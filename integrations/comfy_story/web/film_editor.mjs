// Project recipes are server-owned; this editor never selects or repairs generated takes.
export function filmRunStage(run) {
  if (run.mode === 'first_cut' && run.phase === 'complete') return 'First cut ready to watch and edit'
  const stages = {
    checking_inputs: 'Checking saved inputs',
    checking_verifier: 'Preparing the video reviewer',
    staging_opening: 'Composing the opening image',
    checking_opening: 'Checking the opening image',
    rendering_shot: 'Rendering the current shot',
    rendering_film: 'Rendering the complete first cut',
    checking_shot: 'Reviewing the current shot',
    checking_film: 'Reviewing the complete film',
    complete: 'Generation and machine checks finished',
  }
  return Object.hasOwn(stages, run.phase) ? stages[run.phase] : null
}

export function filmRenderApproach(shot, settings) {
  if (Object.hasOwn(settings, 'opening_prompt')) return 'Stage then animate'
  const profile = settings.render_profile || 'Reference shot'
  const composition = shot.composition || 'Continue frame'
  if (profile === 'Animate frame' && composition === 'Continue frame' && settings.sampler !== 'SPEED Euler 2-stage') return 'Animate starting frame'
  if (profile === 'Reference shot' && composition === 'New composition') return 'Compose from references'
  return 'Custom settings'
}

export function setFilmRenderApproach(shot, settings, approach) {
  if (approach !== 'Stage then animate' && approach !== 'Custom settings') { delete settings.opening_prompt; delete settings.opening_attempts; delete settings.opening_use_previous_scene; delete settings.opening_mode }
  if (approach === 'Stage then animate') {
    shot.composition = 'Continue frame'; settings.render_profile = 'Animate frame'; settings.intent = 'New Scene'
    if (settings.sampler === 'SPEED Euler 2-stage') settings.sampler = 'Native res_multistep'
    settings.opening_prompt ??= ''; settings.opening_attempts ??= 1
  } else if (approach === 'Animate starting frame') {
    if (settings.sampler === 'SPEED Euler 2-stage') throw new Error('Choose Native or Turbo in advanced settings before animating a starting frame.')
    shot.composition = 'Continue frame'; settings.render_profile = 'Animate frame'
  } else if (approach === 'Compose from references') {
    shot.composition = 'New composition'; delete settings.render_profile
  } else if (approach !== 'Custom settings') throw new Error('Choose a supported render approach.')
}

export function validateEditorSeeds(inputs) {
  for (const row of Object.values(inputs.shots_by_id ?? {})) {
    if (!Number.isSafeInteger(row.variation) || row.variation < 0) {
      throw new Error('This seed cannot be represented exactly in the browser. Keep this recipe in the CLI; it has not been changed.')
    }
  }
}

export function parseShotSeconds(value) {
  const milliseconds = Number(value) * 1000
  if (!String(value).trim() || !Number.isSafeInteger(milliseconds) || milliseconds <= 0 ||
      milliseconds > 15000 || milliseconds * 24 % 1000 !== 0) {
    throw new Error('Use 0.125 to 15 seconds in 0.125-second increments for exact video frames.')
  }
  return milliseconds
}

export function filmNodeDuration(values) {
  const suppliedSeconds = Number(String(values['Shot length'] || '10 seconds').split(' ')[0])
  const block = ([5,10,15].includes(suppliedSeconds) ? suppliedSeconds : 10) * 1000
  const cut = values['Output duration (ms)'] ?? 0
  if (!Number.isSafeInteger(cut) || cut < 0 || cut > block) {
    throw new Error('The node output duration must be a whole number of milliseconds within its shot length.')
  }
  return cut === 0 ? block : parseShotSeconds(cut / 1000)
}

export function validateEditorAudio(plan, inputs) {
  const total = plan.shots.reduce((n, shot) => n + shot.duration_ms, 0)
  for (const track of inputs.audio ?? []) {
    if (!Number.isSafeInteger(track.start_ms) || track.start_ms < 0 ||
        !Number.isSafeInteger(track.duration_ms) || track.duration_ms <= 0 ||
        track.start_ms + track.duration_ms > total ||
        !Number.isFinite(track.gain ?? 1) || (track.gain ?? 1) <= 0 || (track.gain ?? 1) > 10) {
      throw new Error('Soundtrack timing must fit the film, with positive duration and volume above 0 and at most 10.')
    }
  }
}

export function validateEditorNarrative(plan) {
  if (plan.narrative == null) return
  const fields = ['goal', 'obstacle', 'decision', 'outcome']
  if (typeof plan.narrative !== 'object' || Object.keys(plan.narrative).length !== fields.length ||
      fields.some(key => typeof plan.narrative[key] !== 'string' || !plan.narrative[key].trim() ||
        plan.narrative[key].length > 2000 || plan.narrative[key].includes('\x00'))) {
    throw new Error('Describe all four intended story fields, or leave them all blank.')
  }
}

export function filmImagePreviewPath(value) {
  if (typeof value !== 'string' || !value || value === 'None' ||
      /[\\\x00-\x1f]/.test(value) || value.includes('://')) return null
  const parts = value.split('/')
  if (parts.some(x => !x || x === '.' || x === '..')) return null
  const filename = parts.pop()
  return '/view?' + new URLSearchParams({type:'input', subfolder:parts.join('/'), filename})
}

const newSeed = () => crypto.getRandomValues(new Uint32Array(1))[0]
// Authoring aid only: these are declared intentions, never inferred video observations.
export function filmStoryboardRows(plan) {
  const state = new Map((plan.initial_facts ?? []).map(fact => [fact.key, fact.value]))
  let start = 0
  return plan.shots.map(shot => {
    const conflicts = (shot.requires ?? []).filter(fact => state.get(fact.key) !== fact.value)
      .map(fact => `${fact.key}: needs ${fact.value}; earlier plan leaves ${state.get(fact.key) ?? 'undeclared'}`)
    const changes = (shot.effects ?? []).map(fact => `${fact.key}: ${state.get(fact.key) ?? 'undeclared'} → ${fact.value}`)
    const row = {shot_id:shot.shot_id, start_ms:start, end_ms:start + shot.duration_ms,
      purpose:shot.purpose, action:shot.action, changes, conflicts}
    for (const fact of shot.effects ?? []) state.set(fact.key, fact.value)
    start = row.end_ms
    return row
  })
}

export function filmCoverageText(run) {
  const c = run.coverage
  if (!c) return null // Older hosts cannot provide reliable duration accounting.
  if (run.mode === 'first_cut') return run.status === 'draft_ready'
    ? `${c.planned_duration_ms / 1000} seconds rendered (${c.planned_shots} shots).`
    : `Planned first cut: ${c.planned_duration_ms / 1000} seconds (${c.planned_shots} shots).`
  const selected = `${c.selected_duration_ms / 1000} of ${c.target_duration_ms / 1000} seconds selected by machine checks (${c.selected_shots}/${c.planned_shots} shots).`
  const incomplete = c.selected_shots < c.planned_shots || c.selected_duration_ms < c.target_duration_ms
  const stopped = ['needs_attention', 'interrupted', 'paused'].includes(run.status)
  return (incomplete && stopped ? 'Incomplete film. ' : '') + selected
}

const names = (text) => text.split(',').map(x => x.trim()).filter(Boolean)
const factsText = (facts = []) => facts.map(x => `${x.key}=${x.value}`).join('\n')
const parseFacts = (text) => text.split('\n').filter(x => x.trim()).map(line => {
  const i = line.indexOf('=')
  if (i < 1 || !line.slice(i + 1).trim()) throw new Error('Facts use one key=value per line.')
  return { key: line.slice(0, i).trim(), value: line.slice(i + 1).trim() }
})
export const parseEndingCounts = (text) => {
  const rows = parseFacts(text).map(({ key, value }) => {
    if (!/^\d+$/.test(value) || Number(value) > 64 || key.length > 120) {
      throw new Error('Ending counts use category=whole number, from 0 to 64, one per line.')
    }
    return { category: key, count: Number(value) }
  })
  if (rows.length > 8 || new Set(rows.map(x => x.category.toLowerCase())).size !== rows.length) {
    throw new Error('Use at most eight distinct ending-count categories.')
  }
  return rows
}

export async function openFilmEditor(api, node, values, document = window.document) {
  const existing = document.getElementById('duet-film-editor')
  if (existing) { existing.focus(); return }
  const dialog = document.createElement('dialog'); dialog.id = 'duet-film-editor'
  dialog.style.cssText = 'width:min(1160px,94vw);max-height:92vh;overflow:auto;background:#171922;color:#eee;border:1px solid #5c5475;border-radius:14px;padding:24px;font:14px system-ui'
  const style = document.createElement('style')
  style.textContent = '#duet-film-editor input,#duet-film-editor textarea,#duet-film-editor select{box-sizing:border-box;width:100%;padding:8px;background:#232635;color:#eee;border:1px solid #514b66;border-radius:6px}#duet-film-editor label{display:block;margin:8px 0}#duet-film-editor button{padding:8px 12px;margin:4px;border:0;border-radius:6px;cursor:pointer}#duet-film-editor section,#duet-film-editor details.duet-film-shot{border:1px solid #494154;padding:14px;margin:12px 0;border-radius:8px}#duet-film-editor details.duet-film-shot>summary{cursor:pointer;font-weight:600}#duet-film-editor video{width:260px;max-width:100%}'
  dialog.append(style)
  const el = (tag, text, parent = dialog) => { const x = document.createElement(tag); if (text) x.textContent = text; parent.append(x); return x }
  const heading = el('h2', 'Comfy Story · Film project')
  el('p', 'Generate → Watch → Edit. Generation runs on the Comfy host and continues when this panel closes.')
  const status = el('p', '')
  status.setAttribute('role', 'status')
  const reportError = error => { status.textContent = error.message ?? String(error) }
  const button = (label, action, parent = dialog) => {
    const b = el('button', label, parent); b.type = 'button'
    b.addEventListener('click', () => Promise.resolve().then(action).catch(reportError)); return b
  }
  const request = async (path, method = 'GET', data) => {
    const response = await api.fetchApi('/duet/story/films' + path, { method, ...(data ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) } : {}) })
    const result = await response.json()
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`)
    return result
  }
  let recipe, run = null, knownRuns = [], dirty = false, pollTimer, newReferenceName = ''
  // Disclosure state belongs to this editor session, never to a film recipe or seed.
  const expandedShots = new Map()
  const expandedControls = new Map()
  const disclosure = (label, key) => {
    const details = document.createElement('details')
    el('summary', label, details)
    details.open = expandedControls.get(key) ?? false
    details.addEventListener('toggle', () => { if (details.isConnected) expandedControls.set(key, details.open) })
    return details
  }
  const soundtrackLevels = new Map()
  const shotViewKey = id => JSON.stringify([recipe.plan.project_id, id])
  const changed = () => { dirty = true; refreshStateCoverage(); refreshStoryboard(); status.textContent = 'Unsaved changes. Save to inspect which shots are affected.' }
  const field = (parent, label, value, onChange, options, affectsRecipe = true) => {
    const wrapper = el('label', label, parent)
    const control = el(options ? 'select' : 'input', '', wrapper)
    if (options) for (const option of options) { const row = el('option', option, control); row.value = option }
    control.value = value ?? ''; control.setAttribute('aria-label', label)
    control.addEventListener(options ? 'change' : 'input', () => { try { onChange(control.value); control.setCustomValidity(''); if (affectsRecipe) changed() } catch (error) { control.setCustomValidity(error.message); changed(); reportError(error) } })
    return control
  }
  const area = (parent, label, value, update) => {
    const wrapper = el('label', label, parent); const control = el('textarea', '', wrapper)
    control.value = value; control.rows = 3; control.setAttribute('aria-label', label)
    control.addEventListener('input', () => { try { update(control.value); control.setCustomValidity(''); changed() } catch (error) { control.setCustomValidity(error.message); changed(); reportError(error) } }); return control
  }
  const imageField = (parent, label, value, update) => {
    const preview = el('figure', '', parent); preview.style.margin = '8px 0'
    const image = el('img', '', preview)
    image.alt = `${label} preview`; image.loading = 'lazy'; image.decoding = 'async'
    image.style.cssText = 'max-width:100%;width:360px;max-height:240px;object-fit:contain'
    const link = el('a', 'Open full image', preview); link.target = '_blank'; link.rel = 'noopener'
    link.setAttribute('aria-label', `Open ${label.toLowerCase()}`)
    const unavailable = el('p', 'Image preview unavailable. Check the uploaded filename.', preview)
    const refresh = file => {
      const path = filmImagePreviewPath(file)
      preview.hidden = !file || file === 'None'
      image.hidden = link.hidden = !path; unavailable.hidden = !!path
      if (path) { image.src = api.apiURL(path); link.href = image.src }
      else { image.removeAttribute('src'); link.removeAttribute('href') }
    }
    image.addEventListener('error', () => { image.hidden = link.hidden = true; unavailable.hidden = false })
    field(parent, label, value, file => { update(file); refresh(file) })
    refresh(value)
  }
  const chooser = el('select', ''); chooser.setAttribute('aria-label', 'Open film project')
  el('option', 'Choose a saved project…', chooser).value = ''
  const actions = el('div', '')
  const projectTools = el('details', '')
  el('summary', 'Project tools and run settings', projectTools)
  const stateCoverage = document.createElement('p')
  stateCoverage.setAttribute('aria-label', 'Declared state checks')
  const refreshStateCoverage = () => {
    const plan = recipe?.plan
    stateCoverage.hidden = !plan
    const hasFacts = plan?.initial_facts?.length || plan?.shots?.some(shot => shot.requires?.length || shot.effects?.length)
    stateCoverage.textContent = !plan ? '' : hasFacts
      ? 'State checks use your declared facts and their visual meanings. They do not verify every detail in Opening composition. Passing a check is a model judgment, not a guarantee.'
      : 'This draft has no explicit starting or ending facts. General action and visibility review can still run, but opening prose does not create state checks. For events that depend on a specific setup, add Required starting facts and Visible ending facts; define their visual meaning in State meanings.'
  }
  const content = el('div', '')
  const storyboard = document.createElement('section')
  storyboard.setAttribute('aria-label', 'Story at a glance')
  const refreshStoryboard = () => {
    storyboard.replaceChildren()
    if (!recipe) return
    el('h3', 'Story at a glance', storyboard)
    el('p', 'Read the sequence before generating. These are planned events, not verified footage. Changes listed here come only from your declared ending facts; prose alone does not add state checks.', storyboard)
    const table = el('table', '', storyboard); table.style.cssText = 'width:100%;border-collapse:collapse;text-align:left'
    const head = el('tr', '', el('thead', '', table))
    for (const label of ['Time', 'Visible event', 'Planned state change']) el('th', label, head).setAttribute('scope', 'col')
    const body = el('tbody', '', table)
    for (const row of filmStoryboardRows(recipe.plan)) {
      const tr = el('tr', '', body)
      el('td', `${row.start_ms / 1000}–${row.end_ms / 1000}s`, tr).style.verticalAlign = 'top'
      const event = el('td', '', tr); event.style.padding = '8px'
      el('strong', row.purpose, event); el('p', row.action, event)
      const change = el('td', '', tr); change.style.padding = '8px'
      for (const text of row.changes.length ? row.changes : ['No ending state change declared.']) el('p', text, change)
      for (const text of row.conflicts) el('p', 'Setup conflict: ' + text, change).setAttribute('role', 'alert')
    }
  }
  const progress = el('section', '')
  let configured = null, projectLoading = true, savePending = false, launchPending = false, importPending = false, audioPending = false
  const projectBusy = () => projectLoading || savePending || launchPending || importPending || audioPending
  const projectPath = () => '/' + encodeURIComponent(recipe.plan.project_id)
  const renderProgress = () => {
    progress.replaceChildren(); el('h3', 'Watch and edit', progress)
    resumeButton.hidden = !run || ['draft_ready', 'ready_for_review'].includes(run.status)
    pauseButton.hidden = run?.status !== 'running' || run?.mode === 'first_cut'
    if (generationMode === 'checked' && configured === null) el('p', 'Save the plan to check the host’s verification setup.', progress)
    if (generationMode === 'checked' && configured === false) el('p', 'The host must configure DUET_STORY_VERIFY_MODEL before verified film generation.', progress)
    if (run) knownRuns = [run, ...knownRuns.filter(x => x.run_id !== run.run_id)]
    if (knownRuns.length) {
      const select = el('select', '', progress); select.setAttribute('aria-label', 'Film run')
      for (const item of knownRuns) { const option = el('option', `${item.status} · recipe ${item.revision.slice(0,8)} · ${item.mode === "first_cut" ? "first cut" : `${item.max_attempts} attempts`}`, select); option.value = item.run_id }
      select.value = run?.run_id || ''
      select.addEventListener('change', () => { clearTimeout(pollTimer); run = knownRuns.find(x => x.run_id === select.value); refreshRun().catch(reportError) })
    }
    if (!run) { el('p', 'No run selected. Save the plan before generating.', progress); return }
    if (run.revision !== recipe.revision) el('p', 'This run belongs to an earlier saved plan. Current edits do not change it.', progress)
    el('p', run.mode === 'first_cut' ? (run.status === 'draft_ready' ? 'First cut ready. Watch it, edit the shots below, save and generate again.' : run.status) : `${run.status} · ${run.shot_id || 'Preparing'} · attempt ${run.attempt || 0}/${run.max_attempts}`, progress)
    const coverage = filmCoverageText(run)
    if (coverage) el('p', coverage, progress)
    const stage = filmRunStage(run)
    if (run.opening_attempt && ['staging_opening', 'checking_opening'].includes(run.phase)) el('p', `Opening image attempt ${run.opening_attempt}`, progress)
    if (stage) el('p', `${run.status === 'running' ? 'Current stage' : 'Last stage'}: ${stage}`, progress)
    if (run.reason) {
      const explanation = run.reason.split(' Recovery record:')[0].trim()
      if (explanation) el('p', explanation, progress)
      const detail = el('details', '', progress)
      el('summary', 'Run details and recovery information', detail)
      el('p', run.reason, detail)
    }
    const openings = [...(run.starting_state_review ? [run.starting_state_review] : []), ...(run.opening_candidates ?? [])]
    for (const candidate of openings) {
      const card = el('section', '', progress)
      el('h4', candidate.kind === 'uploaded' ? 'Starting image check' : `Opening image attempt ${candidate.attempt}`, card)
      if (candidate.unavailable) { el('p', candidate.unavailable, card); continue }
      const path = filmImagePreviewPath(candidate.image)
      if (path) {
        const image = el('img', '', card); image.src = api.apiURL(path)
        image.alt = candidate.kind === 'uploaded' ? 'Assessed starting image' : candidate.fitted ? 'Assessed H3 opening crop' : 'Assessed original opening (legacy run)'
        image.style.maxWidth = '100%'
        const link = el('a', 'Open assessed image', card); link.href = image.src; link.target = '_blank'; link.rel = 'noopener'
      }
      const raw = filmImagePreviewPath(candidate.raw_image)
      if (raw) { const link = el('a', 'Open uncropped source', card); link.href = api.apiURL(raw); link.target = '_blank'; link.rel = 'noopener' }
      el('p', 'Preserved machine assessment. These observations can be wrong; this image is not approved.', card)
      for (const [key, fact] of Object.entries(candidate.state?.conditions ?? {})) {
        el('p', `${key}: expected ${fact.expected}; observed ${fact.observed ?? 'uncertain'}. ${fact.evidence || ''}`, card)
      }
      for (const [name, observation] of Object.entries(candidate.visibility?.observations ?? {})) {
        const expected = candidate.visibility?.requirements?.[name]
        el('p', `${name}: ${expected ? `expected ${expected}; observed ` : ''}${observation.extent}. ${observation.evidence || ''}`, card)
      }
    }
    if (run.pause_requested && run.status === 'running') el('p', 'Pause requested. The current shot will finish first.', progress)
    el('p', run.mode === 'first_cut' ? 'First cuts have not been checked by the visual reviewer.' : 'Machine selections are candidates, not creator approval.', progress)
    if (run.preview) {
      const link = el('a', 'Preview latest candidate film', progress)
      link.href = api.apiURL('/duet/story/films' + projectPath() + '/runs/' + run.run_id + '/preview'); link.target = '_blank'; link.rel = 'noopener'
      el('p', `Through ${run.preview.shot_id} · attempt ${run.preview.attempt}. Includes unapproved footage and uses the chosen film soundtrack. It may be incomplete or rejected.`, progress)
      if (Number.isSafeInteger(run.preview.duration_ms)) el('p', `Candidate preview: ${run.preview.duration_ms / 1000} seconds. Preview length includes the current candidate, even when it failed review.`, progress)
    }
    if ((run.selected ?? []).length) el('p', 'Visual take previews are muted by default. Use the film preview to review the soundtrack.', progress)
    for (const take of [...(run.rendered ?? []), ...(run.selected ?? [])]) {
      const card = el('section', '', progress); el('strong', take.shot_id, card)
      const video = el('video', '', card); video.controls = true; video.preload = 'metadata'; video.muted = true
      video.src = api.apiURL('/duet/story/video/' + encodeURIComponent(take.video_sha256))
    }
    if (['ready_for_review', 'draft_ready'].includes(run.status)) {
      const link = el('a', 'Open assembled film', progress)
      link.href = api.apiURL('/duet/story/films' + projectPath() + '/runs/' + run.run_id + '/video'); link.target = '_blank'; link.rel = 'noopener'
    }
  }
  const refreshRun = async () => {
    if (!run || !dialog.isConnected) return
    const requested = run.run_id
    const result = await request(projectPath() + '/runs/' + requested)
    if (!run || run.run_id !== requested) return
    run = result; renderProgress()
    if (run.status === 'running' && dialog.isConnected) pollTimer = setTimeout(() => refreshRun().catch(reportError), 3000)
  }
  const upload = (parent, label, update) => {
    const input = el('input', '', parent); input.type = 'file'; input.accept = 'image/png,image/jpeg,image/webp'; input.setAttribute('aria-label', label)
    input.addEventListener('change', async () => {
      try {
        if (!input.files[0]) return
        const form = new FormData(); form.append('image', input.files[0]); form.append('type', 'input')
        const response = await api.fetchApi('/upload/image', { method: 'POST', body: form })
        if (!response.ok) throw new Error('Image upload failed')
        const saved = await response.json(); update([saved.subfolder, saved.name].filter(Boolean).join('/')); changed(); render()
      } catch (error) { reportError(error) }
    })
  }
  const render = () => {
    content.replaceChildren(); validateEditorSeeds(recipe.inputs)
    refreshStateCoverage()
    const plan = recipe.plan, inputs = recipe.inputs
    heading.textContent = `Comfy Story · ${plan.title}`
    field(content, 'Film title', plan.title, x => { plan.title = x })
    const narrative = el('details', '', content)
    el('summary', 'Intended story (optional)', narrative)
    if (plan.narrative) narrative.open = true
    el('p', 'Describe the goal, obstacle, decision and outcome viewers should understand. The finished film’s blind retelling is compared with this intent. This does not write the screenplay or guarantee the result. Leave all four blank to omit this comparison.', narrative)
    for (const key of ['goal', 'obstacle', 'decision', 'outcome']) {
      area(narrative, `Intended ${key}`, plan.narrative?.[key] || '', value => {
        plan.narrative ??= {goal:'',obstacle:'',decision:'',outcome:''}
        plan.narrative[key] = value
        if (Object.values(plan.narrative).every(text => !text.trim())) delete plan.narrative
      })
    }
    const filmAdvanced = disclosure('Advanced film settings', JSON.stringify([plan.project_id, 'film']))
    content.append(filmAdvanced)
    filmAdvanced.append(stateCoverage)
    field(filmAdvanced, 'State memory', inputs.recall_selected_state ? 'Selected state evidence' : 'Original references', x => {
      if (x === 'Selected state evidence') inputs.recall_selected_state = true
      else delete inputs.recall_selected_state
    }, ['Original references', 'Selected state evidence'])
    el('p', 'Selected state evidence carries checked appearance changes into later reference shots without creator approval. Animate frame uses its starting image only. Requires the host native Story archive.', filmAdvanced)
    el('p', `${plan.shots.length} shots · ${plan.shots.reduce((n,s) => n+s.duration_ms, 0)/1000} seconds`, content)
    refreshStoryboard(); content.append(storyboard)
    const library = el('section', '', content); el('h3', 'Cast, props and world', library)
    for (const ref of inputs.library.references) {
      const row = el('section', '', library)
      // Existing reference names are IDs; renaming them requires updating the plan explicitly.
      el('strong', ref.name, row)
      field(row, 'Reference role', ref.role, x => { ref.role = x }, ['Character', 'Prop', 'Location', 'Product', 'Style', 'Costume'])
      imageField(row, 'Reference image', ref.file, x => { ref.file = x })
      upload(row, `Upload ${ref.name}`, x => { ref.file = x })
      area(row, 'Baseline appearance', ref.note || '', x => { ref.note = x })
    }
    field(library, 'New reference name', newReferenceName, x => { newReferenceName = x }, undefined, false)
    button('Add reference', () => {
      const name = newReferenceName.trim() || `Reference${inputs.library.references.length + 1}`
      if (!/^[A-Za-z][A-Za-z0-9_-]{0,63}$/.test(name)) throw new Error('Reference names must start with a letter and contain at most 64 letters, numbers, underscores or hyphens.')
      if (inputs.library.references.some(x => x.name.toLowerCase() === name.toLowerCase())) throw new Error('Reference name already exists, ignoring case.')
      inputs.library.references.push({ name, role: 'Character', file: '', note: '' })
      newReferenceName = ''; changed(); render()
    }, library)
    area(filmAdvanced, 'Initial facts (key=value per line)', factsText(plan.initial_facts), x => { plan.initial_facts = parseFacts(x) })
    const meanings = el('details', '', filmAdvanced)
    el('summary', `State meanings (optional) · ${(plan.state_definitions || []).length} definitions`, meanings)
    if (plan.state_definitions?.length) meanings.open = true
    el('p', 'Define what must be visible to establish each state. For any key you define, include all of its planned values. These meanings guide generation and review; they do not establish that an event happened.', meanings)
    for (const [index, meaning] of (plan.state_definitions || []).entries()) {
      const row = el('section', '', meanings)
      field(row, 'State key', meaning.key, x => { meaning.key = x })
      field(row, 'State value', meaning.value, x => { meaning.value = x })
      area(row, 'Observable meaning', meaning.definition, x => { meaning.definition = x })
      button('Remove state meaning', () => {
        plan.state_definitions.splice(index, 1)
        if (!plan.state_definitions.length) delete plan.state_definitions
        changed(); render()
      }, row)
    }
    button('Add state meaning', () => {
      (plan.state_definitions ??= []).push({key:'',value:'',definition:''}); changed(); render()
    }, meanings)
    for (const [index, shot] of plan.shots.entries()) {
      const card = el('details', '', content), settings = inputs.shots_by_id[shot.shot_id]
      card.className = 'duet-film-shot'
      const key = shotViewKey(shot.shot_id)
      if (!expandedShots.has(key)) expandedShots.set(key, index === 0)
      card.open = expandedShots.get(key)
      card.addEventListener('toggle', () => { if (card.isConnected) expandedShots.set(key, card.open) })
      const summary = el('summary', '', card)
      const refreshSummary = () => { summary.textContent = `${index + 1}. ${shot.purpose} · ${shot.duration_ms / 1000} seconds` }
      refreshSummary()
      field(card, 'Purpose', shot.purpose, x => { shot.purpose = x; refreshSummary() })
      area(card, 'Visible action', shot.action, x => { shot.action = x })
      const timing = field(card, 'Seconds', String(shot.duration_ms / 1000), x => { shot.duration_ms = parseShotSeconds(x) })
      timing.inputMode = 'decimal'
      timing.addEventListener('change', () => { if (!timing.validationMessage) render() })
      const advanced = disclosure('Advanced shot settings', JSON.stringify([plan.project_id, shot.shot_id, 'advanced']))
      el('p', 'H3 renders a 5, 10 or 15-second block before trimming to your cut length. Shorter cuts do not guarantee lower generation cost or better action.', advanced)
      field(card, 'Present reference names', (shot.present || []).join(', '), x => { shot.present = names(x) })
      field(advanced, 'Must remain visible throughout', (shot.visible_throughout || []).join(', '), x => { shot.visible_throughout = names(x) })
      field(advanced, 'Must remain fully visible', (shot.fully_visible_throughout || []).join(', '), x => { shot.fully_visible_throughout = names(x) })
      el('p', 'Full visibility also rejects cropping or being hidden behind another subject or object. Leave blank when the shot allows occlusion.', advanced)
      area(advanced, 'Ending visible counts (optional)', (shot.ending_counts || []).map(x => `${x.category}=${x.count}`).join('\n'), x => { shot.ending_counts = parseEndingCounts(x) })
      el('p', 'One category=count per line, for example red mugs=2. Checks only the final frame, including distinguishable partial objects. Blank disables this check. Uncertain counts stop automatic selection; model counting can still be wrong.', advanced)
      field(advanced, 'Absent entities', (shot.absent || []).join(', '), x => { shot.absent = names(x) })
      area(advanced, 'Required starting facts', factsText(shot.requires), x => { shot.requires = parseFacts(x) })
      area(advanced, 'Visible ending facts', factsText(shot.effects), x => { shot.effects = parseFacts(x) })
      field(advanced, 'Depends on shot IDs', (shot.depends_on || []).join(', '), x => { shot.depends_on = names(x) })
      el('small', `Shot ID: ${shot.shot_id}`, advanced)
      const approachControl = field(advanced, 'Render approach', filmRenderApproach(shot, settings), x => {
        if (x === 'Custom settings') { advanced.open = true; return }
        setFilmRenderApproach(shot, settings, x); changed(); render()
      }, ['Animate starting frame', 'Compose from references', 'Stage then animate', 'Custom settings'], false)
      if (Object.hasOwn(settings, 'opening_prompt')) {
        field(advanced, 'Opening image mode', settings.opening_mode || 'Compose', x => { settings.opening_mode = x }, ['Compose', 'Refine'])
        el('p', 'Compose creates a new arrangement from references. Refine is an alpha option for an existing scene with every required subject already placed: it uses a conservative edit to blend the image while trying to retain layout. It cannot reliably add missing subjects or stage a different action. Supply an uploaded scene or a previous selected frame. Both modes remain subject to the same starting-state and visibility checks.', advanced)
        area(advanced, 'Opening composition', settings.opening_prompt, x => { settings.opening_prompt = x })
        field(advanced, 'Opening scene guide', settings.opening_use_previous_scene ? 'Previous selected frame' : 'Uploaded scene or references', x => { if (x === 'Previous selected frame') settings.opening_use_previous_scene = true; else delete settings.opening_use_previous_scene }, ['Uploaded scene or references', 'Previous selected frame'])
        el('p', 'Previous selected frame carries the preceding shot’s actual location into staging. Use it only after the first shot and leave the uploaded starting image blank. It is selected evidence, not creator approval.', advanced)
        field(advanced, 'Maximum opening image attempts', String(settings.opening_attempts ?? 1), x => { settings.opening_attempts = Number(x) }, ['1', '2', '3', '4'])
        el('p', 'Optional Qwen Edit 2511 stage, then H3 animation. Describe the opening separately from the action. Uses up to three sources, including an optional starting image. Starting state and declared visibility are checked before video; fine identity and composition remain fallible. Failed stages are preserved. The first source controls the raw image aspect. Checks and animation use its fitted 1344 × 768 center crop; the raw image is also retained.', advanced)
      }
      el('p', 'Animate starting frame begins from the uploaded scene or previous frame. Stage every needed character and prop in that image. Compose from references requests a new view using the reference roster; it does not preserve the starting composition. On the native Story runtime, its first shot or New Scene can leave the starting image blank to use only selected references. Neither guarantees action, identity, or a cut-free result.', advanced)
      field(advanced, 'Prompt instructions', shot.direction_version === 2 ? 'Positive instructions' : 'Legacy instructions', x => { shot.direction_version = x === 'Positive instructions' ? 2 : 1 }, ['Positive instructions', 'Legacy instructions'])
      el('p', 'Positive instructions describe a continuous view without naming forbidden edit effects. Absent entities remain review requirements but are omitted from generation instructions. Legacy instructions preserve older saved prompts. Changing this setting invalidates the shot and its dependent work; the seed is preserved.', advanced)
      field(advanced, 'Transition', settings.intent || 'New Scene', x => { settings.intent = x }, ['New Scene', 'Next Shot', 'Continue This Shot'])
      field(advanced, 'Composition', shot.composition || 'Continue frame', x => { shot.composition = x; approachControl.value = filmRenderApproach(shot, settings) }, ['Continue frame', 'New composition'])
      field(advanced, 'Camera policy', shot.camera_policy || 'Follow prompt', x => { shot.camera_policy = x; cameraHint.hidden = x !== 'Single continuous shot' }, ['Follow prompt', 'Locked frame', 'Single continuous shot'])
      const cameraHint = el('p', 'Single continuous shot requests one take and checks sampled frame pairs for visible dissolves, ghosted scene layers and wipes. Camera movement is allowed. Unsampled transitions and hard cuts without blending may be missed; this is not a guarantee of uninterrupted motion.', advanced)
      cameraHint.hidden = shot.camera_policy !== 'Single continuous shot'
      field(advanced, 'Frame borders', shot.border_policy || 'Follow prompt', x => { shot.border_policy = x }, ['Follow prompt', 'Preserve opening borders'])
      const setupHint = {
        'Animate starting frame': 'Begin from your starting image, or the previous shot’s closing frame. Every needed character and prop should already be in that image.',
        'Compose from references': 'This shot creates a new view from its selected references. A starting image is optional for the first shot or a new scene.',
        'Stage then animate': 'This shot prepares an opening image from its saved composition before animation. You can review that composition in Advanced shot settings.',
        'Custom settings': 'This shot keeps its saved setup. Advanced shot settings describes its starting-image and reference requirements.',
      }
      el('p', setupHint[filmRenderApproach(shot, settings)], card)
      imageField(card, 'Starting image (blank uses previous frame)', settings.world || '', x => { settings.world = x || null })
      upload(card, 'Upload starting image', x => { settings.world = x })
      imageField(advanced, 'Ending image (optional)', settings.ending_frame || '', x => { if (x) settings.ending_frame = x; else delete settings.ending_frame })
      upload(advanced, 'Upload ending image', x => { settings.ending_frame = x })
      field(advanced, 'Sampler', settings.sampler || 'Native res_multistep', x => { settings.sampler = x; approachControl.value = filmRenderApproach(shot, settings) }, ['Native res_multistep', 'Turbo 4-step', 'Turbo 8-step', 'Full HD 2-pass', 'SPEED Euler 2-stage', 'NVFP4 Exact', 'NVFP4 Balanced', 'NVFP4 Ultra Fast', 'NVFP4 Turbo 4-step'])
      field(advanced, 'Render profile', settings.render_profile || 'Reference shot', x => { if (x === 'Reference shot') delete settings.render_profile; else settings.render_profile = x; approachControl.value = filmRenderApproach(shot, settings) }, ['Reference shot', 'Animate frame'])
      el('p', 'Custom combinations remain available. Reference shot with Continue frame supplies a guide but may reframe or introduce cuts. Use Animate starting frame when the opening composition matters.', advanced)
      field(advanced, 'Prompt format', settings.prompt_format || 'Current', x => { settings.prompt_format = x }, ['Current', 'Structured reference (experimental)', 'H3 automatic v1'])
      field(advanced, 'Locked seed', String(settings.variation), x => {
        const seed = Number(x); if (!x.trim() || !Number.isSafeInteger(seed) || seed < 0) throw new Error('Use an exact nonnegative browser-safe seed')
        settings.variation = seed
      })
      button('New take seed', () => { settings.variation = newSeed(); changed(); render() }, advanced)
      el('p', 'Your seed and rendering settings stay fixed when you save or reopen. Additional controls are optional.', advanced)
      card.append(advanced)
      button('Move earlier', () => { if (index) { [plan.shots[index-1], plan.shots[index]] = [shot, plan.shots[index-1]]; changed(); render() } }, card)
      button('Move later', () => { if (index < plan.shots.length-1) { [plan.shots[index+1], plan.shots[index]] = [shot, plan.shots[index+1]]; changed(); render() } }, card)
      button('Remove shot', () => { plan.shots.splice(index, 1); delete inputs.shots_by_id[shot.shot_id]; expandedShots.delete(key); changed(); render() }, card)
    }
    button('Add shot', () => {
      const id = 'shot-' + crypto.randomUUID().slice(0,8)
      expandedShots.set(shotViewKey(id), true)
      const priorShot = plan.shots.at(-1)
      const prior = priorShot ? inputs.shots_by_id[priorShot.shot_id] : {render_profile:'Animate frame', prompt_format:'H3 automatic v1'}
      plan.shots.push({ shot_id: id, direction_version: priorShot?.direction_version ?? 2, duration_ms: 5000, purpose: 'Next event', action: '', present: [], visible_throughout: [], composition: priorShot?.composition || 'Continue frame' })
      inputs.shots_by_id[id] = { world: null, variation: newSeed(), intent: 'Next Shot', sampler: prior.sampler || 'Native res_multistep' }
      if (prior.prompt_format) inputs.shots_by_id[id].prompt_format = prior.prompt_format
      if (prior.render_profile) inputs.shots_by_id[id].render_profile = prior.render_profile
      changed(); render()
    }, content)
    const sound = el('section', '', content); el('h3', 'Soundtrack', sound)
    const shotSound = el('select', '', sound); shotSound.setAttribute('aria-label','Shot audio')
    for (const label of ['Silent shots', 'Generated speech and effects']) { const option = el('option',label,shotSound); option.value=label }
    shotSound.value = inputs.generated_audio ? 'Generated speech and effects' : 'Silent shots'
    shotSound.addEventListener('change', () => { inputs.generated_audio = shotSound.value === 'Generated speech and effects'; changed() })
    el('p', 'Choose an audio file already uploaded to Comfy, for example through Load Audio. Tracks mix together. Silent shots omit generated audio. Generated speech and effects keeps the model soundtrack and requires H3 automatic prompts. Add exact spoken lines and language to each shot; listen for accuracy. Captions are not added. Soundtrack edits keep visual recipes unchanged. Generate the saved revision to create a new export.', sound)
    for (const [index, track] of (inputs.audio ?? []).entries()) {
      const row = el('section', '', sound)
      el('strong', track.path, row)
      const levels = soundtrackLevels.get(track.sha256)
      if (levels?.status === 'measured') {
        el('p', levels.silent ? 'This audio stream is silent.' : `Audio sample peak: ${levels.peak_dbfs.toFixed(1)} dBFS.`, row)
        if (levels.reaches_full_scale) el('p', 'Audio reaches full scale. Listen for distortion before rendering. Lowering the volume of an already clipped file cannot restore its peaks.', row)
      } else if (levels) el('p', 'Audio levels could not be measured. Listen to the soundtrack before rendering.', row)
      const checkAudio = button('Check audio levels', async () => {
        if (checkAudio.disabled) return
        const target = recipe; checkAudio.disabled = true
        try {
          const asset = await request('/soundtrack', 'POST', { path: track.path })
          if (recipe !== target) throw new Error('The film changed while reading audio; check it again.')
          if (asset.sha256 !== track.sha256) throw new Error('The soundtrack file has changed. Remove it and add the intended file again.')
          soundtrackLevels.set(track.sha256, asset.levels ?? { status: 'unavailable' }); render()
        } finally { checkAudio.disabled = false }
      }, row)
      if (track.cue_id) el('p', `Authored speech cue: ${track.cue_id}`, row)
      const number = (label, value, update) => field(row, label, String(value), x => {
        if (!x.trim() || !Number.isFinite(Number(x))) throw new Error('Enter a finite number')
        update(Number(x))
      })
      number('Soundtrack start (seconds)', track.start_ms / 1000, x => { track.start_ms = Math.round(x * 1000) })
      number('Soundtrack duration (seconds)', track.duration_ms / 1000, x => { track.duration_ms = Math.round(x * 1000) })
      number('Soundtrack volume', track.gain ?? 1, x => { track.gain = x })
      button('Remove soundtrack', () => { inputs.audio.splice(index, 1); changed(); render() }, row)
    }
    let audioName = ''
    field(sound, 'Uploaded soundtrack filename', '', x => { audioName = x.trim() }, undefined, false)
    const addAudio = button('Add soundtrack', async () => {
      if (addAudio.disabled || projectBusy()) return
      if (!audioName) throw new Error('Enter an uploaded Comfy audio filename')
      const target = recipe; addAudio.disabled = true; audioPending = true
      syncProjectActions()
      status.textContent = 'Reading soundtrack duration and levels. Wait for it to finish before saving or generating.'
      try {
        const asset = await request('/soundtrack', 'POST', { path: audioName })
        if (recipe !== target) throw new Error('The film changed while reading audio; choose the soundtrack again.')
        soundtrackLevels.set(asset.sha256, asset.levels ?? { status: 'unavailable' })
        const total = plan.shots.reduce((n, shot) => n + shot.duration_ms, 0)
        inputs.audio ??= []
        inputs.audio.push({ path: asset.path, sha256: asset.sha256, start_ms: 0, duration_ms: Math.min(asset.duration_ms, total), gain: 1 })
        changed(); render()
      } finally { addAudio.disabled = false; audioPending = false; syncProjectActions() }
    }, sound)
    renderProgress()
  }
  const save = async () => {
    if (projectBusy()) return
    const invalid = dialog.querySelector('input:invalid,textarea:invalid,select:invalid')
    if (invalid) {
      for (let parent = invalid.parentElement; parent && parent !== dialog; parent = parent.parentElement) {
        if (parent.tagName === 'DETAILS') parent.open = true
      }
      invalid.reportValidity(); throw new Error('Correct the highlighted field before saving.')
    }
    validateEditorSeeds(recipe.inputs)
    validateEditorAudio(recipe.plan, recipe.inputs)
    validateEditorNarrative(recipe.plan)
    savePending = true; syncProjectActions()
    try {
      if (!recipe.inputs.library.project_name?.trim()) recipe.inputs.library.project_name = recipe.plan.title.trim()
      recipe.plan.target_duration_ms = recipe.plan.shots.reduce((n,s) => n+s.duration_ms, 0)
      const result = await request(recipe.revision ? projectPath() : '', recipe.revision ? 'PUT' : 'POST', { plan: recipe.plan, inputs: recipe.inputs, expected_revision: recipe.revision || null })
      recipe = result.recipe; dirty = false; node.properties ??= {}; node.properties.duet_film_project_id = recipe.plan.project_id
      const option = Array.from(chooser.children).find(x => x.value === recipe.plan.project_id) || el('option', '', chooser)
      option.value = recipe.plan.project_id; option.textContent = recipe.plan.title; chooser.value = option.value
      const detail = await request(projectPath()); configured = detail.verification_configured
      status.textContent = result.impact ? `Saved. Reusable prefix: ${result.impact.reusable_prefix.join(', ') || 'none'}. Recheck: ${result.impact.requires_generation_review.join(', ') || 'none'}.` : 'Saved immutable film recipe.'
      if (result.impact?.narrative_review_changed) status.textContent += ' Whole-film review is required for the changed story intent.'
      render()
    } finally { savePending = false; syncProjectActions() }
  }
  const newFilmButton = button('New film', () => {
    if (projectBusy()) return
    if (dirty) throw new Error('Save the current draft before starting another film.')
    clearTimeout(pollTimer)
    const id = 'film-' + crypto.randomUUID(), shotId = 'shot-1'
    recipe = {
      plan: {project_id:id, title:'New film', target_duration_ms:5000, initial_facts:[], shots:[{shot_id:shotId, direction_version:2, duration_ms:5000, purpose:'Opening', action:'', present:[], composition:'Continue frame'}]},
      inputs: {library:{project_name:'', references:[]}, shots_by_id:{[shotId]:{world:null, variation:newSeed(), sampler:'Native res_multistep', render_profile:'Animate frame', prompt_format:'H3 automatic v1'}}, burn_subtitles:false},
    }
    run = null; knownRuns = []; configured = null; chooser.value = ''; attempts = 2; budgetControl.value = '2'; generationMode = 'first_cut'; modeControl.value = 'First cut'; newReferenceName = ''
    changed(); render()
    status.textContent = 'New unsaved film. Add your story and references, then save. Previous projects and host runs are unchanged.'
  }, actions)
  newFilmButton.disabled = true
  const saveButton = button('Save plan', save, actions)
  let attempts = 2, generationMode = 'first_cut'
  const modeControl = field(projectTools, 'Generation mode', 'First cut', x => { generationMode = x === 'First cut' ? 'first_cut' : 'checked'; renderProgress() }, ['First cut', 'Check each shot'], false)
  el('p', 'First cut renders the full film for you to watch and edit. Check each shot requires a visual reviewer and may stop before the film is complete. Attempts apply to checked runs.', projectTools)
  const budgetControl = field(projectTools, 'Maximum attempts per shot', '2', x => { attempts = Number(x) }, ['1','2','3','4'], false)
  const launch = async resume => {
    if (projectBusy()) return
    if (resume && !run) throw new Error('Choose an existing run to resume.')
    if (!resume && (dirty || !recipe.revision)) throw new Error('Save and inspect the affected shots before generating.')
    const path = projectPath()
    const payload = resume ? { revision: run.revision, max_attempts: run.max_attempts, run_id: run.run_id, mode: run.mode || 'checked' } : { revision: recipe.revision, max_attempts: attempts, mode: generationMode }
    launchPending = true
    syncProjectActions()
    status.textContent = 'Checking model and input files before starting. This can take several minutes. Your request is pending; do not submit it again.'
    try {
      run = await request(path + '/runs', 'POST', payload)
      status.textContent = 'Run started on the host. You can close this panel and return later.'
      renderProgress(); await refreshRun()
    } finally {
      launchPending = false
      syncProjectActions()
    }
  }
  const generateButton = button('Generate saved plan', () => launch(false), actions)
  const resumeButton = button('Resume selected run', () => launch(true), actions)
  const pauseButton = button('Pause after current shot', async () => {
    if (projectBusy()) return
    if (!run) throw new Error('No active run')
    run = await request(projectPath() + '/runs/' + run.run_id + '/pause', 'POST', {}); renderProgress()
  }, actions)
  const downloadBundleButton = button('Download inputs bundle', () => {
    if (projectBusy()) return
    if (dirty || !recipe.revision) throw new Error('Save the plan before exporting its inputs.')
    const link = document.createElement('a'); link.href = api.apiURL('/duet/story/films' + projectPath() + '/inputs-bundle?revision=' + recipe.revision); link.download = 'duet-story-inputs.zip'; link.click()
  }, projectTools)
  const bundleUpload = el('input', '', projectTools); bundleUpload.type = 'file'; bundleUpload.accept = '.zip'; bundleUpload.hidden = true
  const importBundleButton = button('Import inputs bundle', () => { if (!projectBusy()) bundleUpload.click() }, projectTools)
  bundleUpload.addEventListener('change', async () => {
    if (projectBusy()) return
    importPending = true; syncProjectActions()
    try {
      const file = bundleUpload.files?.[0]; if (!file) return
      const body = new FormData(); body.append('bundle', file)
      const response = await api.fetchApi('/duet/story/films/import-inputs', {method:'POST', body})
      const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Inputs bundle import failed')
      node.properties ??= {}; node.properties.duet_film_project_id = result.recipe.plan.project_id
      await load(result.recipe.plan.project_id); status.textContent = `Imported ${result.assets_restored} input assets. No completed-take approvals were imported.`
    } catch (error) { reportError(error) } finally { bundleUpload.value = ''; importPending = false; syncProjectActions() }
  })
  const downloadRecipeButton = button('Download recipe', () => {
    if (projectBusy()) return
    validateEditorSeeds(recipe.inputs)
    const blob = new Blob([JSON.stringify(recipe, null, 2)], {type:'application/json'})
    const url = URL.createObjectURL(blob), a = el('a', '')
    a.href = url; a.download = `${recipe.plan.project_id}.json`; a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000)
    status.textContent = 'Recipe downloaded. Media and model assets are not included; retain the story store backup.'
  }, projectTools)
  const importFile = el('input', '', projectTools); importFile.type = 'file'; importFile.accept = '.json'; importFile.setAttribute('aria-label', 'Import film recipe')
  importFile.addEventListener('change', async () => {
    if (projectBusy()) return
    importPending = true; syncProjectActions()
    try {
      if (!importFile.files[0]) return
      if (importFile.files[0].size > 2*1024*1024) throw new Error('Recipe exceeds 2 MiB')
      const imported = JSON.parse(await importFile.files[0].text()); validateEditorSeeds(imported.inputs)
      imported.plan.project_id = 'film-' + crypto.randomUUID(); imported.revision = null
      recipe = imported; run = null; knownRuns = []; changed(); render()
    } catch(error) { reportError(error) }
    finally { importFile.value = ''; importPending = false; syncProjectActions() }
  })
  function syncProjectActions() {
    const pending = projectBusy()
    for (const control of [chooser, newFilmButton, saveButton, generateButton, resumeButton, budgetControl, modeControl, pauseButton, downloadBundleButton, importBundleButton, downloadRecipeButton, bundleUpload, importFile]) control.disabled = pending
    content.inert = progress.inert = pending
  }
  const load = async id => {
    clearTimeout(pollTimer)
    projectLoading = true; syncProjectActions()
    status.textContent = 'Loading the selected film project…'
    try {
      const data = await request('/' + encodeURIComponent(id)); validateEditorSeeds(data.recipe.inputs)
      node.properties ??= {}; node.properties.duet_film_project_id = id; chooser.value = id
      recipe = data.recipe; configured = data.verification_configured; knownRuns = data.runs; run = data.runs[0] || null; if (run) { attempts = run.max_attempts; budgetControl.value = String(attempts) } dirty = false; render(); await refreshRun()
      status.textContent = 'Opened saved film project.'
    } catch (error) {
      chooser.value = node.properties?.duet_film_project_id || ''; throw error
    } finally { projectLoading = false; syncProjectActions() }
  }
  chooser.addEventListener('change', async () => {
    if (projectBusy()) return
    if (dirty) { chooser.value = node.properties?.duet_film_project_id || ''; status.textContent = 'Save the current draft before opening another project.'; return }
    if (chooser.value) { try { await load(chooser.value) } catch (error) { reportError(error) } }
  })
  button('Close', () => dialog.close(), actions)
  dialog.addEventListener('close', () => { clearTimeout(pollTimer); dialog.remove() })
  syncProjectActions()
  document.body.append(dialog); dialog.showModal()
  try {
    const saved = await request('')
    for (const item of saved.projects) { const option = el('option', item.title, chooser); option.value = item.project_id }
    if (node.properties?.duet_film_project_id) await load(node.properties.duet_film_project_id)
    else {
      let library = JSON.parse(values['Story Library'] || '[]')
      if (Array.isArray(library) && library.length === 0) library = {project_name:'',references:[]}
      if (!Array.isArray(library.references)) throw new Error('Set up the Story reference library first.')
      const id = 'film-' + crypto.randomUUID(), shotId = 'shot-1'
      const action = values['What happens next?'] || ''
      const duration = filmNodeDuration(values)
      recipe = {plan: {project_id:id,title:library.project_name || 'New story',target_duration_ms:duration,initial_facts:[],shots:[{shot_id:shotId,direction_version:2,duration_ms:duration,purpose:'Opening',action,present:library.references.filter(x=>action.includes('@'+x.name)).map(x=>x.name),composition:values.Composition || 'Continue frame'}]},
        inputs: {library,shots_by_id:{[shotId]:{world:values['World / starting frame'] === 'None' ? null : (values['World / starting frame'] || null),variation:values.Variation ?? 0,sampler:values.Sampler || 'Native res_multistep'}},burn_subtitles:false}}
      if (values['Render profile'] && values['Render profile'] !== 'Reference shot') recipe.inputs.shots_by_id[shotId].render_profile = values['Render profile']
      if (!action.trim() && !library.references.length && !library.project_name &&
          (!values.Composition || values.Composition === 'Continue frame') &&
          (!values.Sampler || values.Sampler === 'Native res_multistep') &&
          (!values['Render profile'] || values['Render profile'] === 'Reference shot')) recipe.inputs.shots_by_id[shotId].render_profile = 'Animate frame'
      if (values['Prompt format'] && values['Prompt format'] !== 'Current') recipe.inputs.shots_by_id[shotId].prompt_format = values['Prompt format']
      dirty = false; render()
    }
  } catch (error) { reportError(error) }
  finally { projectLoading = false; syncProjectActions() }
}
