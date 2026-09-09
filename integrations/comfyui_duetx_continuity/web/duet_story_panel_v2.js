import { api } from '../../scripts/api.js'
import { app } from '../../scripts/app.js'
import { openFilmEditor } from './film_editor.mjs'
import { CURRENT_WIDGET_ORDER, deriveStoryView, encodeLibrary, encodeMemoryCommands, fieldMapFromWidgets, isDuetStoryNode, keepAsCurrentLook, migrateSerializedStoryWidgets, nextShotWiring, normalizeInspectorSummary, normalizeLibrary, pendingMemoryCommands, recordCompletedStory, setPendingMemoryCommands, stageMemoryCommand } from './story_state.mjs?v=7'

const CURRENT_NODE_TYPE = 'DuetStory'
const CANON_NODE_TYPE = 'DuetStoryCanon'
const OPTIONAL_DEFAULTS = {'Composition':'Continue frame', 'Output duration (ms)':0, 'Render profile':'Reference shot', 'Prompt format':'Current'}
const SAVED_WIDGET_ORDER = [...CURRENT_WIDGET_ORDER, ...Object.keys(OPTIONAL_DEFAULTS)]
const NAMED_CONTINUITY_FIELDS = ['Scene entities', 'Shot state evidence']
const ROLES = ['Character', 'Product', 'Prop', 'Location', 'Costume', 'Style']

function installStyles() {
  if (document.getElementById('duet-story-style')) return
  const style = document.createElement('style')
  style.id = 'duet-story-style'
  style.textContent = `
    .duet-story{box-sizing:border-box;min-width:390px;height:100%;overflow:auto;padding:16px;border:1px solid #514a78;border-radius:15px;background:linear-gradient(145deg,#242034,#12141d);color:#f7f4ff;font:13px/1.42 Inter,system-ui,sans-serif}.duet-story *{box-sizing:border-box}.duet-story-head,.duet-story-row,.duet-story-actions{display:flex;align-items:center;gap:9px}.duet-story-head{justify-content:space-between;margin-bottom:6px}.duet-story-title{font-size:16px;font-weight:760}.duet-story-state{padding:3px 8px;border-radius:999px;background:#273e3a;color:#a9ead7;font-size:11px}.duet-story-copy{margin:0 0 13px;color:#bdb7cc}.duet-story-help{margin:5px 0 0;color:#918ba3;font-size:11px}.duet-story-label{display:block;margin:11px 0 5px;color:#9992aa;font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase}.duet-story input,.duet-story select,.duet-story textarea{width:100%;border:1px solid #4c465e;border-radius:7px;padding:7px 8px;background:#171922;color:#f5f2ff}.duet-story textarea{resize:vertical;min-height:90px}.duet-story-settings{margin-top:12px}.duet-story-library{max-height:200px;overflow:auto;display:grid;gap:7px}.duet-story-reference{padding:9px;border:1px solid #3b374b;border-radius:9px;background:#1a1b26}.duet-story-reference .duet-story-row{display:grid;grid-template-columns:1fr 110px 28px}.duet-story-file{color:#b9afe9;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.duet-story-chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}.duet-story-chip{padding:4px 9px!important;border:1px solid #40385c!important;border-radius:999px!important;background:#252334;color:#8d879d;font-weight:600!important}.duet-story-chip[data-active=true]{background:#30294b;color:#dacfff}.duet-story-chip[data-protected=true]{border-color:#59aa8f!important;background:#24443b;color:#c8ffec}.duet-story-motion{margin-top:7px;padding:6px 8px;border-radius:7px;background:#191b25;color:#918ba3}.duet-story-motion[data-connected=true]{color:#9fe4ca}.duet-story-status{margin-top:10px;color:#9fe4ca}.duet-story-status[data-ready=false]{color:#ffbd8e}.duet-story button{border:0;border-radius:8px;padding:8px 10px;cursor:pointer;font-weight:700}.duet-story button:disabled{opacity:.45;cursor:not-allowed}.duet-story-primary{flex:1;background:#8f7cff;color:white}.duet-story-next{flex:1;background:#2e544d;color:#d5fff4}.duet-story-secondary{background:#302d3d;color:#d8d1e9}.duet-story-remove{width:28px;padding:6px!important;background:#412c36;color:#ffc4cf}.duet-story-actions{margin-top:13px}.duet-story-upload{display:none}.duet-story-inspector{display:grid;gap:6px}.duet-story-memory-row{padding:8px;border:1px solid #3b374b;border-radius:8px;background:#171922}.duet-story-memory-title{font-weight:700}.duet-story-memory-note{color:#bdb7cc}.duet-story-memory-buttons{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}.duet-story-memory-buttons button{padding:5px 7px;font-size:11px}`
  document.head.appendChild(style)
}

const findWidget = (node, name) => (node.widgets ?? []).find((item) => item.name === name)

function values(node) {
  const mapped = fieldMapFromWidgets(node.widgets ?? [])
  for (const name of ['Create', 'Story Library', 'Reference context', 'What happens next?', 'Keep this detail', 'Shot length', 'Variation', 'Story revision', 'Reference policy', 'Memory backend', 'Sampler', 'Memory actions', 'Composition', 'Output duration (ms)', 'Render profile', 'Prompt format']) {
    const item = findWidget(node, name)
    if (item) mapped[name] = item.value
  }
  return mapped
}

function setWidget(node, name, value) {
  const item = findWidget(node, name)
  if (!item) return
  item.value = value
  item.callback?.(value)
  node.graph?.setDirtyCanvas?.(true, true)
}

const hasInputLink = (node, name, fallback) => {
  const input = (node.inputs ?? []).find((item) => item.name === name) ?? node.inputs?.[fallback]
  return input?.link != null
}

const connected = (node) => ({
  story: hasInputLink(node, 'Previous Story', 0),
  frame: hasInputLink(node, 'Previous Frame', 1),
  motion: hasInputLink(node, 'Motion reference', -1),
  starting: hasInputLink(node, 'Starting image', -1),
})

async function uploadReference(file) {
  const form = new FormData()
  form.append('image', file)
  form.append('type', 'input')
  form.append('overwrite', 'false')
  const response = await api.fetchApi('/upload/image', { method: 'POST', body: form })
  if (!response.ok) throw new Error(`Upload failed (${response.status})`)
  const result = await response.json()
  return [result.subfolder, result.name].filter(Boolean).join('/')
}

function createNextShot(node) {
  const next = globalThis.LiteGraph?.createNode?.(CURRENT_NODE_TYPE)
  if (!next) throw new Error('Comfy could not create the next Comfy Story node')
  app.graph.add(next)
  next.pos = [(node.pos?.[0] ?? 0) + (node.size?.[0] ?? 440) + 90, node.pos?.[1] ?? 0]
  const current = values(node)
  setWidget(next, 'Create', 'Next Shot')
  setWidget(next, 'Story Library', current['Story Library'])
  setWidget(next, 'Reference context', current['Reference context'])
  setWidget(next, 'Keep this detail', current['Keep this detail'])
  setWidget(next, 'Shot length', current['Shot length'])
  setWidget(next, 'Reference policy', current['Reference policy'])
  setWidget(next, 'Memory actions', encodeMemoryCommands(pendingMemoryCommands(node)))
  setWidget(next, 'Memory backend', current['Memory backend'])
  setWidget(next, 'Sampler', current.Sampler)
  setWidget(next, 'Composition', current.Composition ?? 'Continue frame')
  setWidget(next, 'Render profile', current['Render profile'] ?? 'Reference shot')
  setWidget(next, 'Output duration (ms)', current['Output duration (ms)'] ?? 0)
  setWidget(next, 'Prompt format', current['Prompt format'] ?? 'Current')
  nextShotWiring(node, next)
  node.properties ??= {}
  node.properties.duet_pending_memory = []
  node.__duetRefresh?.()
  app.canvas?.selectNode?.(next)
  app.canvas?.centerOnNode?.(next)
}

function attachEditor(node) {
  if (node.__duetStoryEditor || typeof node.addDOMWidget !== 'function') return
  installStyles()
  if (node.properties?.duet_story_inspector && !node.__duetInspector) {
    try { recordCompletedStory(node, node.properties.duet_story_inspector) }
    catch (error) { console.error('Rejected saved Duet inspector', error) }
  }
  const livingCanon = true
  for (const name of [...SAVED_WIDGET_ORDER, 'Keep this detail', 'Compiler experiment']) {
    const item = findWidget(node, name)
    if (item) { item.hidden = true; item.computeSize = () => [0, 0] }
  }
  const root = document.createElement('section')
  root.className = 'duet-story'
  root.innerHTML = `<div class="duet-story-head"><span class="duet-story-title">Comfy Story</span><span class="duet-story-state"></span></div><p class="duet-story-copy">Build the next shot. Your cast, world, and important earlier moments travel through the green Story State wire.</p><label class="duet-story-label">Story project</label><input class="duet-story-project" placeholder="Name this story"/><label class="duet-story-label">Reference context</label><select class="duet-story-context"><option>Native</option><option>Compiled preview</option></select><p class="duet-story-help">Native H3 is the validated generation path. Compiled preview is experimental; quality and speed advantages are unproven.</p><div class="duet-story-motion" data-connected="false">Motion reference · not connected</div><div class="duet-story-head"><span class="duet-story-label">Shared reference library</span><button class="duet-story-secondary duet-story-add" type="button">+ Add reference</button></div><div class="duet-story-library"></div><input class="duet-story-upload" type="file" accept="image/png,image/jpeg,image/webp"/><div class="duet-story-chips"></div><p class="duet-story-help">Select references with @Name. Named references and approved details share the image budget; the final set is checked before rendering. After a shot, approve a visible moment to carry its look forward.</p><div class="duet-story-status"></div><div class="duet-story-actions"><button class="duet-story-primary" type="button">Run workflow</button><button class="duet-story-next" type="button">Add Next Shot →</button></div>`
  const shotControls = document.createElement('div')
  const advanced = document.createElement('details')
  advanced.className = 'duet-story-settings'
  const advancedTitle = document.createElement('summary')
  advancedTitle.textContent = 'Generation settings and recovery'
  advanced.append(advancedTitle)
  const controls = new Map()
  const addField = (container, name, label, options = null, multiline = false) => {
    const wrapper = document.createElement('label')
    const heading = document.createElement('span')
    heading.className = 'duet-story-label'; heading.textContent = label
    const control = document.createElement(options ? 'select' : multiline ? 'textarea' : 'input')
    control.setAttribute('aria-label', label)
    if (options) control.replaceChildren(...options.map((value) => {
      const option = document.createElement('option'); option.value = value; option.textContent = value; return option
    }))
    if (['Variation', 'Output duration (ms)'].includes(name)) { control.type = 'number'; control.min = '0'; control.max = String(Number.MAX_SAFE_INTEGER); control.step = '1' }
    control.value = findWidget(node, name)?.value ?? ''
    control.addEventListener(options ? 'change' : 'input', () => {
      if (['Variation', 'Output duration (ms)'].includes(name) && (!Number.isSafeInteger(Number(control.value)) || Number(control.value) < 0)) return
      setWidget(node, name, ['Variation', 'Output duration (ms)'].includes(name) ? Number(control.value) : control.value)
    })
    wrapper.append(heading, control); container.append(wrapper); controls.set(name, control)
  }
  addField(shotControls, 'Create', 'Create', ['Start Story', 'Continue This Shot', 'Next Shot', 'New Scene'])
  addField(shotControls, 'What happens next?', 'What happens next?', null, true)
  addField(shotControls, 'Render profile', 'Render profile', ['Reference shot', 'Animate frame'])
  const profileHelp = document.createElement('p'); profileHelp.className = 'duet-story-help'
  profileHelp.textContent = 'Reference shot uses selected reference images. Animate frame animates the starting frame with the first-frame model; it requires Continue frame and does not recall separate images.'
  shotControls.append(profileHelp)
  addField(shotControls, 'Shot length', 'Shot length', ['5 seconds', '10 seconds', '15 seconds'])
  addField(shotControls, 'World / starting frame', 'World / starting frame', null)
  const worldHelp = document.createElement('p'); worldHelp.className = 'duet-story-help'
  worldHelp.textContent = 'Use an uploaded filename or choose an image. With None, Next Shot uses the connected Previous Frame; Frame animation and Continue frame require a starting image for a new story or scene. Native Reference shot with New composition can start from selected references alone.'
  shotControls.append(worldHelp)
  const worldUpload = document.createElement('input'); worldUpload.type = 'file'; worldUpload.accept = 'image/png,image/jpeg,image/webp'; worldUpload.hidden = true
  const worldButton = document.createElement('button'); worldButton.type = 'button'; worldButton.className = 'duet-story-secondary'; worldButton.textContent = 'Choose starting image…'
  worldButton.addEventListener('click', () => worldUpload.click())
  worldUpload.addEventListener('change', async () => {
    const file = worldUpload.files?.[0]
    if (file) {
      try { setWidget(node, 'World / starting frame', await uploadReference(file)) }
      catch (error) { status.textContent = error.message; status.dataset.ready = 'false' }
    }
    worldUpload.value = ''
  })
  shotControls.append(worldButton, worldUpload)
  addField(advanced, 'Variation', 'Variation')
  addField(advanced, 'Reference policy', 'Reference policy', ['Automatic', 'Prompt mentions only'])
  addField(advanced, 'Sampler', 'Sampler', ['Native res_multistep', 'SPEED Euler 2-stage', 'Turbo 4-step', 'Turbo 8-step', 'Full HD 2-pass', 'NVFP4 Exact', 'NVFP4 Balanced', 'NVFP4 Ultra Fast', 'NVFP4 Turbo 4-step'])
  addField(advanced, 'Composition', 'Composition', ['Continue frame', 'New composition'])
  addField(advanced, 'Output duration (ms)', 'Output duration (ms)')
  addField(advanced, 'Prompt format', 'Prompt format', ['Current', 'Structured reference (experimental)', 'H3 automatic v1'])
  addField(advanced, 'Story revision', 'Recovery revision JSON', null, true)
  root.querySelector('.duet-story-copy').after(shotControls)
  root.querySelector('.duet-story-actions').before(advanced)
  const project = root.querySelector('.duet-story-project')
  const libraryRoot = root.querySelector('.duet-story-library')
  const upload = root.querySelector('.duet-story-upload')
  const context = root.querySelector('.duet-story-context')
  const motion = root.querySelector('.duet-story-motion')
  const sound = document.createElement('div')
  sound.className = 'duet-story-motion'
  sound.setAttribute('aria-label', 'Shot audio source')
  const soundHelp = document.createElement('p')
  soundHelp.className = 'duet-story-help'
  soundHelp.textContent = 'For exact narration, connect this shot’s approved track to Authored audio. It replaces generated sound; it does not animate lip sync.'
  motion.after(sound, soundHelp)
  const chips = root.querySelector('.duet-story-chips')
  const status = root.querySelector('.duet-story-status')
  const state = root.querySelector('.duet-story-state')
  const generate = root.querySelector('.duet-story-primary')
  root.querySelector('.duet-story-title').textContent = 'Comfy Story'
  const inspector = document.createElement('section')
  inspector.className = 'duet-story-inspector'
  if (livingCanon) status.after(inspector)
  const preview = document.createElement('video')
  preview.controls = true; preview.preload = 'metadata'; preview.hidden = true
  preview.style.width = '100%'; preview.setAttribute('aria-label', 'Accepted shot video')
  inspector.before(preview)
  let library = normalizeLibrary(findWidget(node, 'Story Library')?.value)
  if (library.error && !library.project_name) library = { project_name: '', references: [], error: null }
  let pendingReference = null
  const save = () => setWidget(node, 'Story Library', encodeLibrary(library))
  const stagedCommands = () => pendingMemoryCommands(node)
  const stageAction = (command) => {
    const next = stageMemoryCommand(stagedCommands(), command)
    setPendingMemoryCommands(node, next)
    renderInspector()
  }
  const actionButton = (label, action) => {
    const button = document.createElement('button')
    button.type = 'button'; button.className = 'duet-story-secondary'; button.textContent = label
    button.addEventListener('click', action)
    return button
  }
  root.querySelector('.duet-story-head').append(actionButton('Open Story', () => openFilmEditor(api, node, values(node))))
  const memoryButton = (label, command) => actionButton(label, () => stageAction(command))
  const currentLookButton = (label, options) => actionButton(label, () => {
    try {
      const next = keepAsCurrentLook(stagedCommands(), options)
      setPendingMemoryCommands(node, next)
      status.textContent = `${label} staged for the next shot`
      status.dataset.ready = 'true'
      renderInspector()
    } catch (error) {
      status.textContent = error.message
      status.dataset.ready = 'false'
    }
  })
  const memorySection = (label) => {
    const section = document.createElement('section')
    const heading = document.createElement('span'); heading.className = 'duet-story-label'; heading.textContent = label
    section.append(heading); return section
  }
  const memoryRow = (title, note = '') => {
    const row = document.createElement('div'); row.className = 'duet-story-memory-row'
    const heading = document.createElement('div'); heading.className = 'duet-story-memory-title'; heading.textContent = title
    row.append(heading)
    if (note) { const detail = document.createElement('div'); detail.className = 'duet-story-memory-note'; detail.textContent = note; row.append(detail) }
    return row
  }
  const renderInspector = () => {
    if (!livingCanon) return
    const summary = node.__duetInspector
    preview.hidden = !summary?.video_sha256
    if (summary?.video_sha256 && preview.dataset.digest !== summary.video_sha256) {
      preview.src = api.apiURL(`/duet/story/video/${summary.video_sha256}`)
      preview.dataset.digest = summary.video_sha256
    }
    inspector.replaceChildren()
    if (!summary) {
      const empty = memoryRow('No authenticated Story State yet', 'Generate the first shot to begin story memory.')
      inspector.append(empty); return
    }
    const parent = summary.revision_sha256
    const used = memorySection('Used for this shot')
    for (const binding of summary.used_for_this_shot) used.append(memoryRow(binding.role.replaceAll('-', ' ')))
    const canon = memorySection('Story Canon')
    for (const entity of summary.canon) {
      const row = memoryRow(entity.reference_name, `${entity.state_note || 'Approved original'} · ${entity.presence.replace('_', ' ')}`)
      const buttons = document.createElement('div'); buttons.className = 'duet-story-memory-buttons'
      buttons.append(memoryButton('Restore original', { action: 'restore_original', parent_revision_sha256: parent, target_id: entity.entity_id }))
      for (const [label, presence] of [['Present next shot', 'present'], ['Off screen next shot', 'off_screen']]) {
        buttons.append(memoryButton(label, {
          action: 'update_canon', parent_revision_sha256: parent, target_id: entity.entity_id,
          presence, state_note: entity.state_note || 'Original appearance',
          supporting_evidence_ids: entity.supporting_evidence_ids,
        }))
      }
      row.append(buttons); canon.append(row)
    }
    for (const pending of summary.pending) {
      const row = memoryRow(`Review ${pending.entity_id}`, pending.state_note)
      const buttons = document.createElement('div'); buttons.className = 'duet-story-memory-buttons'
      buttons.append(currentLookButton('Keep as current look', {
        parentRevisionSha256: parent,
        entityId: pending.entity_id,
        evidenceId: pending.evidence_id,
        presence: 'unknown',
        stateNote: pending.state_note,
      }))
      row.append(buttons); canon.append(row)
    }
    const moments = memorySection('Important Moments')
    for (const moment of summary.important_moments.slice(-12).reverse()) {
      const location = moment.shot_index == null ? 'legacy evidence' : `shot ${moment.shot_index + 1} · frame ${moment.frame_index + 1}`
      const row = memoryRow(moment.entity_ids.join(', ') || moment.kind, location)
      const preview = document.createElement('img')
      preview.src = api.apiURL(`/duet/story/evidence/${moment.asset_sha256}`)
      preview.alt = `Stored evidence from ${location}`
      preview.loading = 'lazy'
      preview.style.cssText = 'width:100%;max-height:180px;object-fit:contain;border-radius:6px;margin-top:6px'
      row.append(preview)
      const buttons = document.createElement('div'); buttons.className = 'duet-story-memory-buttons'
      buttons.append(
        memoryButton(moment.retained ? 'Let fade' : 'Keep this detail', { action: moment.retained ? 'let_fade' : 'keep', parent_revision_sha256: parent, target_id: moment.evidence_id }),
        memoryButton('Use in next shot', { action: 'use_in_this_shot', parent_revision_sha256: parent, target_id: moment.evidence_id }),
        memoryButton('Forget', { action: 'forget', parent_revision_sha256: parent, target_id: moment.evidence_id }),
      )
      for (const entityId of moment.entity_ids) {
        const entity = summary.canon.find((item) => item.entity_id === entityId)
        const reference = entity?.reference_name ?? entityId
        buttons.append(currentLookButton(`Keep @${reference} look`, {
          parentRevisionSha256: parent,
          entityId,
          evidenceId: moment.evidence_id,
          presence: 'unknown',
          stateNote: `Creator-confirmed visual state from ${location}`,
        }))
      }
      row.append(buttons); moments.append(row)
    }
    const changes = stagedCommands()
    if (changes.length) {
      const draft = memoryRow(`${changes.length} changes for the next shot`, 'Add Next Shot to apply them. This completed take stays unchanged.')
      draft.append(actionButton('Clear staged changes', () => { setPendingMemoryCommands(node, []); renderInspector() }))
      inspector.append(draft)
    }
    inspector.append(used, canon, moments)
  }
  let refresh
  const renderRows = () => {
    libraryRoot.replaceChildren(...library.references.map((reference, index) => {
      const row = document.createElement('div'); row.className = 'duet-story-reference'
      const top = document.createElement('div'); top.className = 'duet-story-row'
      const name = document.createElement('input'); name.placeholder = 'Reference name'; name.value = reference.name
      const role = document.createElement('select')
      role.replaceChildren(...ROLES.map((value) => { const option = document.createElement('option'); option.value = value; option.textContent = value; return option }))
      role.value = reference.role
      const remove = document.createElement('button'); remove.className = 'duet-story-remove'; remove.type = 'button'; remove.textContent = '×'
      const note = document.createElement('input'); note.placeholder = 'What must stay recognizable?'; note.value = reference.note
      const file = document.createElement('button'); file.className = 'duet-story-secondary duet-story-file'; file.type = 'button'; file.textContent = reference.file || 'Choose image…'
      name.addEventListener('input', () => { reference.name = name.value; save(); refresh() })
      role.addEventListener('change', () => { reference.role = role.value; save(); refresh() })
      note.addEventListener('input', () => { reference.note = note.value; save() })
      file.addEventListener('click', () => { pendingReference = index; upload.click() })
      remove.addEventListener('click', () => { library.references.splice(index, 1); save(); renderRows(); refresh() })
      top.append(name, role, remove); row.append(top, note, file); return row
    }))
  }
  refresh = () => {
    const view = deriveStoryView(values(node), connected(node), { allowNoMentions: livingCanon })
    for (const [name, control] of controls) {
      if (document.activeElement !== control) control.value = findWidget(node, name)?.value ?? ''
    }
    project.value = view.library.project_name || library.project_name
    const inherited = connected(node).story
    worldHelp.textContent = connected(node).starting
      ? 'Starting image is connected. Start Story and New Scene use that image instead of the uploaded filename. Next Shot continues to use Previous Frame when connected.'
      : 'Use an uploaded filename, choose an image, or connect an image node to Starting image. With None, Next Shot uses Previous Frame; Frame animation and Continue frame require a starting image for a new story or scene. Native Reference shot with New composition can start from selected references alone.'
    project.disabled = inherited
    root.querySelector('.duet-story-add').disabled = inherited
    for (const control of libraryRoot.querySelectorAll('input, select, button')) control.disabled = inherited
    libraryRoot.title = inherited ? 'This library is inherited from the connected Story State.' : ''
    context.value = view.referenceContext
    motion.dataset.connected = String(view.motion)
    motion.textContent = `Motion reference · ${view.motion ? 'connected' : 'not connected'}`
    const authored = hasInputLink(node, 'Authored audio', -1)
    sound.dataset.connected = String(authored)
    sound.textContent = authored ? 'Audio · approved track connected' : 'Audio · generated sound'
    state.textContent = livingCanon && node.__duetInspector ? `${node.__duetInspector.shot_count} shots remembered` : view.state
    status.textContent = view.message
    status.dataset.ready = String(view.ready)
    const active = new Set(view.active)
    const protectedNames = new Set(view.protected)
    chips.replaceChildren(...view.allReferences.map((reference) => {
      const chip = document.createElement('button')
      chip.type = 'button'; chip.className = 'duet-story-chip'; chip.dataset.active = String(active.has(reference)); chip.dataset.protected = String(protectedNames.has(reference))
      chip.textContent = `${reference}${protectedNames.has(reference) ? ' · keep exact' : ''}`
      chip.disabled = livingCanon || !active.has(reference)
      chip.title = livingCanon
        ? 'Living Canon manages exact evidence in the memory inspector'
        : active.has(reference) ? 'Toggle exact detail preservation' : 'Mention this reference in the prompt to activate it'
      chip.addEventListener('click', () => {
        const next = new Set(view.protected)
        if (next.has(reference)) next.delete(reference)
        else if (next.size < 2) next.add(reference)
        setWidget(node, 'Keep this detail', [...next].join(', ')); refresh()
      })
      return chip
    }))
    generate.disabled = !view.ready
    renderInspector()
  }
  project.addEventListener('input', () => { library.project_name = project.value; save(); refresh() })
  context.addEventListener('change', () => { setWidget(node, 'Reference context', context.value); refresh() })
  root.querySelector('.duet-story-add').addEventListener('click', () => { library.references.push({ file: '', name: `Reference${library.references.length + 1}`, note: '', role: 'Character' }); save(); renderRows(); refresh() })
  upload.addEventListener('change', async () => {
    const file = upload.files?.[0]
    if (file && pendingReference != null) {
      try { library.references[pendingReference].file = await uploadReference(file); save(); renderRows(); refresh() }
      catch (error) { status.textContent = error.message; status.dataset.ready = 'false' }
    }
    upload.value = ''; pendingReference = null
  })
  generate.addEventListener('click', async () => { generate.disabled = true; try { await app.queuePrompt?.(0) } finally { refresh() } })
  const reroll = actionButton('New take', async () => {
    const old = Number(findWidget(node, 'Variation')?.value ?? 0)
    setWidget(node, 'Variation', Number.isSafeInteger(old) && old < Number.MAX_SAFE_INTEGER ? old + 1 : Math.floor(Math.random() * 0xFFFFFFFF))
    await app.queuePrompt?.(0)
  })
  root.querySelector('.duet-story-actions').append(reroll)
  root.querySelector('.duet-story-next').addEventListener('click', () => createNextShot(node))
  for (const item of node.widgets ?? []) {
    const previous = item.callback
    item.callback = function (...args) { const result = previous?.apply(this, args); setWidget(this, 'Prompt format', 'H3 automatic v1'); queueMicrotask(refresh); return result }
  }
  node.addDOMWidget('duet_story_editor', 'Comfy Story', root, { getHeight: () => 800, getMinHeight: () => 600, getValue: () => 'duet-story', hideOnZoom: false, setValue: () => {}, serialize: false })
  node.__duetStoryEditor = root
  node.__duetRefresh = refresh
  node.setSize?.([Math.max(node.size?.[0] ?? 0, 450), Math.max(node.size?.[1] ?? 0, 900)])
  renderRows(); refresh()
}

app.registerExtension({
  name: 'duet.story',
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (![CURRENT_NODE_TYPE, CANON_NODE_TYPE].includes(nodeData?.name)) return
    if (!nodeType.prototype.__duetWidgetMigrationInstalled) {
      const previousSerialize = nodeType.prototype.onSerialize
      nodeType.prototype.onSerialize = function (info) {
        const result = previousSerialize?.call(this, info)
        const fields = Object.fromEntries(SAVED_WIDGET_ORDER.map((name) => [name, findWidget(this, name)?.value ?? OPTIONAL_DEFAULTS[name]]))
        for (const name of NAMED_CONTINUITY_FIELDS) {
          const widget = findWidget(this, name)
          if (widget) fields[name] = widget.value
        }
        info.properties ??= {}
        info.properties.duet_story_fields = fields
        info.widgets_values = SAVED_WIDGET_ORDER.map((name) => fields[name])
        return result
      }
      const previousConfigure = nodeType.prototype.onConfigure
      nodeType.prototype.onConfigure = function (info) {
        const original = info?.widgets_values
        const savedFields = info?.properties?.duet_story_fields
        const migrated = savedFields && CURRENT_WIDGET_ORDER.every((name) => Object.hasOwn(savedFields, name))
          ? SAVED_WIDGET_ORDER.map((name) => savedFields[name] ?? OPTIONAL_DEFAULTS[name])
          : migrateSerializedStoryWidgets(nodeData.name, original)
        const changed = Array.isArray(original) && (migrated.length !== original.length || migrated.some((value, index) => value !== original[index]))
        if (changed) info.widgets_values = migrated
        const result = previousConfigure?.call(this, info)
        const restore = () => {
          if (Array.isArray(migrated) && [CURRENT_WIDGET_ORDER.length, CURRENT_WIDGET_ORDER.length + 1, SAVED_WIDGET_ORDER.length].includes(migrated.length)) {
            for (const [index, name] of SAVED_WIDGET_ORDER.entries()) {
              const widget = findWidget(this, name)
              if (widget) widget.value = index < migrated.length ? migrated[index] : OPTIONAL_DEFAULTS[name]
            }
            for (const name of NAMED_CONTINUITY_FIELDS) {
              const widget = findWidget(this, name)
              if (widget && savedFields && Object.hasOwn(savedFields, name)) widget.value = savedFields[name]
            }
            this.__duetRefresh?.()
          }
        }
        restore()
        queueMicrotask(restore)
        return result
      }
      nodeType.prototype.__duetWidgetMigrationInstalled = true
    }
    const previous = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function (...args) { const result = previous?.apply(this, args); setWidget(this, 'Prompt format', 'H3 automatic v1'); queueMicrotask(() => attachEditor(this)); return result }
  },
  nodeCreated(node) { if (isDuetStoryNode(node)) queueMicrotask(() => attachEditor(node)) },
})

api.addEventListener('executed', (event) => {
  const value = event?.detail?.output?.duet_story
  if (!value) return
  try {
    const summary = normalizeInspectorSummary(value)
    const node = app.graph?.getNodeById?.(summary.owner_node_id)
    if (!node || !isDuetStoryNode(node)) return
    recordCompletedStory(node, summary)
    if (node.__duetRefresh) queueMicrotask(node.__duetRefresh)
  } catch (error) {
    console.error('Rejected Comfy Story inspector payload', error)
  }
})
