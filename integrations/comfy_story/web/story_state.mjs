const REFERENCE_NAME = /^[A-Za-z][A-Za-z0-9_-]*$/
const ROLES = new Set(['Character', 'Product', 'Prop', 'Location', 'Costume', 'Style'])
const DIGEST = /^[0-9a-f]{64}$/
const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
const MEMORY_ACTIONS = new Set(['keep', 'use_in_this_shot', 'update_canon', 'restore_original', 'let_fade', 'forget'])
const PRESENCE = new Set(['present', 'off_screen', 'unknown'])

export const CURRENT_WIDGET_ORDER = Object.freeze([
  'Create',
  'Story Library',
  'World / starting frame',
  'What happens next?',
  'Shot length',
  'Variation',
  'Story revision',
  'Reference policy',
  'Sampler',
  'Memory actions',
])

export const FIELD_ORDER = Object.freeze([
  'Previous Story',
  'Previous Frame',
  ...CURRENT_WIDGET_ORDER,
  'Composition',
  'Output duration (ms)',
  'Render profile',

  'Prompt format',
])

export function fieldMapFromWidgets(widgets) {
  return Object.fromEntries(
    widgets.map((widget, index) => [FIELD_ORDER.includes(widget.name) ? widget.name : FIELD_ORDER[index], widget.value]),
  )
}

export function restoreSerializedStoryWidgets(values) {
  if (!Array.isArray(values)) return values
  if (values.length < CURRENT_WIDGET_ORDER.length || values.length > CURRENT_WIDGET_ORDER.length + 4
      || !['5 seconds', '10 seconds', '15 seconds'].includes(values[4])
      || !Number.isSafeInteger(values[5]) || values[5] < 0) {
    throw new Error('Unsupported Comfy Story widget layout')
  }
  return [...values]
}

export function isComfyStoryNode(node) {
  const identities = [
    node?.comfyClass,
    node?.type,
    node?.title,
    node?.constructor?.comfyClass,
    node?.constructor?.type,
    node?.constructor?.nodeData?.name,
    node?.constructor?.nodeData?.display_name,
  ]
  return identities.some((identity) => ['ComfyStory', 'Comfy Story'].includes(identity))
}

function exactObject(value, fields, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`)
  const keys = Object.keys(value).sort()
  if (keys.join('|') !== [...fields].sort().join('|')) throw new Error(`${label} has missing or unknown fields`)
  return value
}

function text(value, label, maximum = 1000) {
  if (typeof value !== 'string' || value.length > maximum) throw new Error(`${label} must be bounded text`)
  return value
}

function identifier(value, label) {
  if (typeof value !== 'string' || !IDENTIFIER.test(value)) throw new Error(`${label} must be a portable identifier`)
  return value
}

function digest(value, label) {
  if (typeof value !== 'string' || !DIGEST.test(value)) throw new Error(`${label} must be a lowercase digest`)
  return value
}

function stringArray(value, label) {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`)
  return value.map((item) => identifier(item, label))
}

export function normalizeInspectorSummary(value) {
  // ComfyUI aggregates each node UI field as a list, including dynamic subgraphs.
  if (Array.isArray(value) && value.length === 1) value = value[0]
  const fields = ['canon', 'important_moments', 'owner_node_id', 'pending', 'revision_sha256', 'shot_count', 'used_for_this_shot']
  if (value && Object.hasOwn(value, 'video_sha256')) fields.push('video_sha256')
  const root = exactObject(value, fields, 'Story inspector summary')
  if (!Array.isArray(root.canon) || !Array.isArray(root.important_moments) || !Array.isArray(root.pending) || !Array.isArray(root.used_for_this_shot)) throw new Error('Story inspector sections must be arrays')
  if (!Number.isInteger(root.shot_count) || root.shot_count < 0 || root.shot_count > 128) throw new Error('Story inspector shot count is invalid')
  const canon = root.canon.map((raw) => {
    const row = exactObject(raw, ['entity_id', 'presence', 'reference_name', 'state_note', 'supporting_evidence_ids'], 'Canon row')
    if (!PRESENCE.has(row.presence)) throw new Error('Canon presence is invalid')
    return {
      entity_id: identifier(row.entity_id, 'Canon entity'),
      presence: row.presence,
      reference_name: text(row.reference_name, 'Canon reference', 96),
      state_note: text(row.state_note, 'Canon state', 500),
      supporting_evidence_ids: stringArray(row.supporting_evidence_ids, 'Canon evidence'),
    }
  })
  const important_moments = root.important_moments.map((raw) => {
    const row = exactObject(raw, ['asset_sha256', 'entity_ids', 'evidence_id', 'frame_index', 'kind', 'remembered', 'retained', 'salience_q', 'selected', 'shot_index'], 'Important moment')
    for (const key of ['frame_index', 'shot_index']) if (row[key] !== null && (!Number.isInteger(row[key]) || row[key] < 0)) throw new Error(`Important moment ${key} is invalid`)
    if (!Number.isInteger(row.salience_q)) throw new Error('Important moment salience is invalid')
    for (const key of ['remembered', 'retained', 'selected']) if (typeof row[key] !== 'boolean') throw new Error(`Important moment ${key} is invalid`)
    return {
      asset_sha256: digest(row.asset_sha256, 'Important moment asset'),
      entity_ids: stringArray(row.entity_ids, 'Important moment entities'),
      evidence_id: identifier(row.evidence_id, 'Important moment evidence'),
      frame_index: row.frame_index,
      kind: text(row.kind, 'Important moment kind', 32),
      remembered: row.remembered,
      retained: row.retained,
      salience_q: row.salience_q,
      selected: row.selected,
      shot_index: row.shot_index,
    }
  })
  const pending = root.pending.map((raw) => {
    const row = exactObject(raw, ['entity_id', 'evidence_id', 'source_revision_sha256', 'state_note'], 'Pending canon row')
    return {
      entity_id: identifier(row.entity_id, 'Pending entity'),
      evidence_id: identifier(row.evidence_id, 'Pending evidence'),
      source_revision_sha256: digest(row.source_revision_sha256, 'Pending source'),
      state_note: text(row.state_note, 'Pending state', 500),
    }
  })
  const used_for_this_shot = root.used_for_this_shot.map((raw) => {
    const row = exactObject(raw, ['asset_sha256', 'role'], 'Used guide')
    return { asset_sha256: digest(row.asset_sha256, 'Used guide asset'), role: identifier(row.role, 'Used guide role') }
  })
  return {
    ...(Object.hasOwn(root, 'video_sha256') ? { video_sha256: digest(root.video_sha256, 'Accepted video') } : {}),
    canon,
    important_moments,
    owner_node_id: identifier(String(root.owner_node_id), 'Owner node'),
    pending,
    revision_sha256: digest(root.revision_sha256, 'Story revision'),
    shot_count: root.shot_count,
    used_for_this_shot,
  }
}

function normalizeMemoryCommand(raw) {
  const action = text(raw?.action, 'Memory action', 32)
  if (!MEMORY_ACTIONS.has(action)) throw new Error('Memory action is unsupported')
  const presence = raw?.presence == null ? null : text(raw.presence, 'Memory presence', 32)
  if (presence !== null && !PRESENCE.has(presence)) throw new Error('Memory presence is unsupported')
  return {
    action,
    parent_revision_sha256: digest(raw?.parent_revision_sha256, 'Memory parent revision'),
    presence,
    state_note: text(raw?.state_note ?? '', 'Memory state', 500),
    supporting_evidence_ids: stringArray(raw?.supporting_evidence_ids ?? [], 'Memory support'),
    target_id: identifier(raw?.target_id, 'Memory target'),
  }
}

export function stageMemoryCommand(commands, command) {
  if (!Array.isArray(commands)) throw new Error('Staged memory commands must be an array')
  const normalized = normalizeMemoryCommand(command)
  const existing = commands.map(normalizeMemoryCommand)
  if (existing.some((item) => item.parent_revision_sha256 !== normalized.parent_revision_sha256)) throw new Error('Staged actions must use one parent revision')
  return [...existing.filter((item) => item.target_id !== normalized.target_id), normalized]
}

export function keepAsCurrentLook(commands, {
  parentRevisionSha256,
  entityId,
  evidenceId,
  stateNote,
  presence = 'unknown',
}) {
  const note = text(stateNote, 'Current look', 500).trim()
  if (!note) throw new Error('Describe the current look before keeping it')
  const retained = stageMemoryCommand(commands, {
    action: 'keep',
    parent_revision_sha256: parentRevisionSha256,
    target_id: evidenceId,
  })
  return stageMemoryCommand(retained, {
    action: 'update_canon',
    parent_revision_sha256: parentRevisionSha256,
    presence,
    state_note: note,
    supporting_evidence_ids: [evidenceId],
    target_id: entityId,
  })
}

export function encodeMemoryCommands(commands) {
  if (!Array.isArray(commands) || commands.length > 64) throw new Error('At most 64 memory actions may be staged')
  return JSON.stringify(commands.map(normalizeMemoryCommand))
}

export function pendingMemoryCommands(node) {
  return JSON.parse(encodeMemoryCommands(node.properties?.comfy_pending_memory ?? []))
}

export function setPendingMemoryCommands(node, commands) {
  const normalized = JSON.parse(encodeMemoryCommands(commands))
  const revision = node.__comfyInspector?.revision_sha256
  if (!revision || normalized.some((item) => item.parent_revision_sha256 !== revision)) throw new Error('Pending changes must target the displayed completed shot')
  node.properties ??= {}
  node.properties.comfy_pending_memory = normalized
  node.graph?.setDirtyCanvas?.(true, true)
}

export function recordCompletedStory(node, value) {
  const summary = normalizeInspectorSummary(value)
  node.properties ??= {}
  const pending = pendingMemoryCommands(node).filter((item) => item.parent_revision_sha256 === summary.revision_sha256)
  node.properties.comfy_pending_memory = pending
  node.properties.comfy_story_inspector = summary
  node.__comfyInspector = summary
  // Generating widgets are immutable inputs to this completed take. Keeping
  // them unchanged lets its request identity recover the same output on restart.
  return summary
}

export function extractMentions(prompt) {
  const result = []
  const seen = new Set()
  const pattern = /(^|[^A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_-]*)/g
  for (const match of String(prompt ?? '').matchAll(pattern)) {
    const name = match[2]
    const key = name.toLocaleLowerCase()
    if (!seen.has(key)) {
      seen.add(key)
      result.push(name)
    }
  }
  return result
}

export function normalizeLibrary(value) {
  let parsed
  try {
    parsed = typeof value === 'string' ? JSON.parse(value || '{}') : value
  } catch {
    return { project_name: '', references: [], error: 'Story Library JSON is invalid' }
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { project_name: '', references: [], error: 'Story Library is missing' }
  }
  const projectName = String(parsed.project_name ?? '').trim()
  if (!projectName) return { project_name: '', references: [], error: 'Name this story project' }
  if (!Array.isArray(parsed.references) || parsed.references.length === 0) {
    return { project_name: projectName, references: [], error: 'Add at least one reference' }
  }
  const references = []
  const seen = new Set()
  for (const raw of parsed.references) {
    const reference = {
      name: String(raw?.name ?? '').trim(),
      role: String(raw?.role ?? ''),
      note: String(raw?.note ?? '').trim(),
      file: String(raw?.file ?? '').trim(),
    }
    const key = reference.name.toLocaleLowerCase()
    if (!REFERENCE_NAME.test(reference.name)) {
      return { project_name: projectName, references, error: 'Reference names use letters, numbers, _ or -' }
    }
    if (seen.has(key)) return { project_name: projectName, references, error: `Duplicate @${reference.name}` }
    if (!ROLES.has(reference.role)) return { project_name: projectName, references, error: `Choose a role for @${reference.name}` }
    if (!reference.file) return { project_name: projectName, references, error: `Choose a file for @${reference.name}` }
    seen.add(key)
    references.push(reference)
  }
  return { project_name: projectName, references, error: null }
}

export function encodeLibrary(library) {
  return JSON.stringify({
    project_name: String(library.project_name ?? '').trim(),
    references: (library.references ?? []).map(({ name, role, note, file }) => ({ file, name, note, role })),
  })
}

export function normalizeProtectedNames(value, availableNames, selectedNames = availableNames) {
  const available = new Map(availableNames.map((name) => [name.toLocaleLowerCase(), name]))
  const selected = new Set(selectedNames.map((name) => name.toLocaleLowerCase()))
  const result = []
  const seen = new Set()
  for (const raw of String(value ?? '').split(',')) {
    const requested = raw.trim().replace(/^@/, '').trim()
    if (!requested) continue
    const key = requested.toLocaleLowerCase()
    const canonical = available.get(key)
    if (!canonical) throw new Error(`@${requested} is not in the Story Library`)
    if (!selected.has(key)) throw new Error(`@${canonical} must be selected in this shot`)
    if (!seen.has(key)) {
      seen.add(key)
      result.push(canonical)
    }
  }
  if (result.length > 2) throw new Error('Keep this detail supports at most two references')
  return result
}

export function deriveStoryView(values, connected = {}, options = {}) {
  if (Object.hasOwn(values, 'Reference context') && values['Reference context'] !== 'Native') {
    throw new Error('Unsupported reference context; Comfy Story uses native H3 references')
  }
  const library = normalizeLibrary(values['Story Library'])
  const names = library.references.map((reference) => reference.name)
  const lookup = new Map(names.map((name) => [name.toLocaleLowerCase(), name]))
  const mentions = extractMentions(values['What happens next?'])
  const unknown = mentions.filter((name) => !lookup.has(name.toLocaleLowerCase()))
  const known = [...new Set(mentions.map((name) => lookup.get(name.toLocaleLowerCase())).filter(Boolean))]
  const motion = Boolean(connected.motion)
  const allowNoMentions = Boolean(options.allowNoMentions)
  // A two-packet recall budget is not a two-name cast limit. The server
  // additionally checks the exact expanded identity/state image roster.
  const maxNamedReferences = motion ? 7 : 8
  let protectedNames = []
  let protectionError = null
  try {
    protectedNames = normalizeProtectedNames(values['Keep this detail'], names, known)
  } catch (error) {
    protectionError = error.message
  }
  let message = `${known.length} active reference${known.length === 1 ? '' : 's'}`
  if (library.error) message = library.error
  else if (mentions.length === 0 && !allowNoMentions) message = 'Mention a reference, like @Maya'
  else if (mentions.length === 0) message = 'No named references selected for this shot'
  else if (unknown.length) message = `Unknown: ${unknown.map((name) => `@${name}`).join(', ')}`
  else if (known.length > maxNamedReferences) message = `Use at most ${maxNamedReferences} named references in this shot`
  else if (protectionError) message = protectionError
  const ready = !library.error
    && (allowNoMentions || mentions.length > 0)
    && unknown.length === 0
    && known.length <= maxNamedReferences
    && !protectionError
  const inherited = Boolean(connected.story)
  const frameReady = Boolean(connected.frame)
  const intent = String(values.Create ?? 'Start Story')
  const state = inherited ? 'Inherited memory' : intent === 'Start Story' ? 'New story' : 'Connect Previous Story'
  return {
    active: known.map((name) => `@${name}`),
    allReferences: names.map((name) => `@${name}`),
    protected: protectedNames.map((name) => `@${name}`),
    motion,
    intent,
    library,
    message,
    ready: ready && (intent === 'Start Story' || inherited) && (intent !== 'Continue This Shot' || frameReady),
    state,
  }
}

export function nextShotWiring(source, target) {
  if (!source || !target || typeof source.connect !== 'function') throw new Error('Comfy Story nodes are required')
  source.connect(2, target, 0)
  source.connect(1, target, 1)
  return [
    { from: 2, to: 0, type: 'COMFY_STORY' },
    { from: 1, to: 1, type: 'IMAGE' },
  ]
}
