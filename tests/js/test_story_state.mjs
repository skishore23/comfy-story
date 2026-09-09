import assert from 'node:assert/strict'
import test from 'node:test'

import {
  deriveStoryView,
  encodeLibrary,
  extractMentions,
  fieldMapFromWidgets,
  isComfyStoryNode,
  restoreSerializedStoryWidgets,
  nextShotWiring,
  normalizeLibrary,
} from '../../integrations/comfy_story/web/story_state.mjs'

const library = {
  project_name: 'Coast Story',
  references: [
    { file: 'maya.png', name: 'Maya', note: 'red scarf', role: 'Character' },
    { file: 'chest.png', name: 'WoodenChest', note: 'marked lid', role: 'Prop' },
    { file: 'coast.png', name: 'Coast', note: '', role: 'Location' },
  ],
}

test('a character, prop and setting fit without confusing names with state packets', () => {
  assert.deepEqual(normalizeLibrary(encodeLibrary(library)), { ...library, error: null })
  const options = { allowNoMentions: true }
  assert.equal(deriveStoryView({ Create: 'Start Story', 'Story Library': encodeLibrary(library), 'What happens next?': '@Maya opens @WoodenChest' }, {}, options).ready, true)
  const selected = deriveStoryView({ Create: 'Start Story', 'Story Library': encodeLibrary(library), 'What happens next?': '@Maya carries @WoodenChest toward @Coast' }, {}, options)
  assert.equal(selected.ready, true)
  assert.equal(selected.message, '3 active references')
})

test('mention parsing is case-insensitive and unknown names are actionable', () => {
  assert.deepEqual(extractMentions('@Maya sees @Ghost and @maya'), ['Maya', 'Ghost'])
  const view = deriveStoryView({ Create: 'Start Story', 'Story Library': encodeLibrary(library), 'What happens next?': '@Ghost arrives' })
  assert.equal(view.ready, false)
  assert.equal(view.message, 'Unknown: @Ghost')
})

test('next shot requires inherited state and continue also requires frame', () => {
  const values = { Create: 'Continue This Shot', 'Story Library': encodeLibrary(library), 'What happens next?': '@Maya walks' }
  assert.equal(deriveStoryView(values, {}).ready, false)
  assert.equal(deriveStoryView(values, { story: true }).ready, false)
  assert.equal(deriveStoryView(values, { story: true, frame: true }).ready, true)
})

test('Living Canon automatic policy permits a shot without explicit mentions', () => {
  const values = { Create: 'Start Story', 'Story Library': encodeLibrary(library), 'What happens next?': 'An empty corridor' }
  assert.equal(deriveStoryView(values, {}, { allowNoMentions: true }).ready, true)
})

test('Add Next Shot connects shared state and last frame', () => {
  const calls = []
  const source = { connect: (...args) => calls.push(args) }
  const target = {}
  assert.deepEqual(nextShotWiring(source, target), [
    { from: 2, to: 0, type: 'COMFY_STORY' },
    { from: 1, to: 1, type: 'IMAGE' },
  ])
  assert.deepEqual(calls, [[2, target, 0], [1, target, 1]])
})

test('Nodes 2.0 widget order remains a fallback and identity is exact', () => {
  const fields = fieldMapFromWidgets([
    { name: 'widget_0', value: null }, { name: 'widget_1', value: null },
    { name: 'widget_2', value: 'Next Shot' }, { name: 'widget_3', value: encodeLibrary(library) },
  ])
  assert.equal(fields.Create, 'Next Shot')
  assert.equal(isComfyStoryNode({ comfyClass: 'ComfyStory' }), true)
  assert.equal(isComfyStoryNode({ title: 'Comfy Story' }), true)
  assert.equal(isComfyStoryNode({ title: 'Comfy Story' }), true)
  assert.equal(isComfyStoryNode({ type: 'ComfyStory' }), true)
  assert.equal(isComfyStoryNode({ title: 'Comfy Story (legacy workflow alias)' }), false)
  assert.equal(isComfyStoryNode({ title: 'Comfy Story (legacy workflow alias)' }), false)
  assert.equal(isComfyStoryNode({ comfyClass: 'ComfyStoryCanon' }), false)
  assert.equal(isComfyStoryNode({ title: 'Comfy Continuity' }), false)
})

test('unsupported saved layouts are rejected without positional drift', () => {
  const old = ['Next Shot', 'library', 'world.png', '@Maya runs', '10 seconds', 7, 'rev', 'Automatic']
  const compiled = ['Next Shot', 'library', 'Compiled preview', 'world.png', '@Maya runs', 'Maya', '10 seconds', 7, 'rev', 'Automatic', 'MiniMax H3', 'SPEED Euler 2-stage']
  for (const values of [old,compiled]) {
    const before=[...values]
    assert.throws(() => restoreSerializedStoryWidgets(values), /Unsupported.*layout/)
    assert.deepEqual(values,before)
  }
})

test('saved current widget layout retains sampler and memory actions', () => {
  const values = ['Next Shot', 'library', 'world.png', '@Maya runs', '5 seconds', 9, 'rev', 'Prompt mentions only', 'SPEED Euler 2-stage', '[{"action":"keep"}]']
  const restored = restoreSerializedStoryWidgets(values)
  assert.deepEqual(restored, values)
  assert.notEqual(restored, values)
})

test('new composition survives save/load without being mistaken for the historical twelve-field layout', () => {
  const fields = ['Start Story', '{}', 'world.png', '@Maya turns', '10 seconds', 12, '', 'Automatic', 'Native res_multistep', '[]', 'New composition']
  assert.deepEqual(restoreSerializedStoryWidgets(fields), fields)
  assert.equal(fieldMapFromWidgets([{ name: 'Composition', value: 'New composition' }]).Composition, 'New composition')
})

test('public panel still enforces the image ceiling with and without motion', () => {
  const many = { project_name: 'Roster', references: Array.from({length: 9}, (_, i) => ({name: `Ref${i}`, role: 'Character', file: `ref-${i}.png`, note: ''})) }
  const fields = count => ({Create: 'Start Story', 'Story Library': encodeLibrary(many), 'What happens next?': many.references.slice(0, count).map(r=>`@${r.name}`).join(' ')})
  const options = {allowNoMentions: true}
  assert.equal(deriveStoryView(fields(8), {}, options).ready, true)
  assert.equal(deriveStoryView(fields(9), {}, options).ready, false)
  assert.equal(deriveStoryView(fields(7), {motion:true}, options).ready, true)
  assert.equal(deriveStoryView(fields(8), {motion:true}, options).ready, false)
})
