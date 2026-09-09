import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import * as state from '../../integrations/comfy_story/web/story_state.mjs'

// Execute the actual panel lifecycle; helper-only tests miss serialization hooks.
const source = readFileSync(new URL('../../integrations/comfy_story/web/story_panel.js', import.meta.url), 'utf8').replace(/^import .*\n/gm, '')
const names = [...state.CURRENT_WIDGET_ORDER, 'Composition', 'Output duration (ms)', 'Render profile', 'Prompt format']
function fixture() {
  let extension
  let next
  const context = vm.createContext({
    ...state,
    app: { registerExtension: value => { extension = value }, graph: { add: value => { next = value } } },
    api: { addEventListener() {} },
    queueMicrotask: fn => fn(),
    console,
  })
  vm.runInContext(source, context)
  class Node {
    constructor() {
      this.widgets = names.map(name => ({ name, value: name === 'Composition' ? 'Continue frame' : name === 'Prompt format' ? 'Current' : name === 'Output duration (ms)' ? 0 : name }))
      this.properties = {}
      this.links = []
    }
    connect(...args) { this.links.push(args) }
  }
  extension.beforeRegisterNodeDef(Node, { name: 'DuetStory' })
  context.LiteGraph = { createNode: () => new Node() }
  return { Node, context, next: () => next }
}
const widget = (node, name) => node.widgets.find(item => item.name === name)

test('actual panel saves and restores composition with named and positional fields', () => {
  const { Node } = fixture()
  const original = new Node()
  widget(original, 'Composition').value = 'New composition'
  widget(original, 'Output duration (ms)').value = 5000
  widget(original, 'Render profile').value = 'Animate frame'
  const info = {}
  original.onSerialize(info)
  assert.equal(info.widgets_values.length, 15)
  assert.equal(info.properties.duet_story_fields.Composition, 'New composition')
  for (const saved of [info, { widgets_values: info.widgets_values }]) {
    const restored = new Node()
    restored.onConfigure(saved)
    assert.equal(widget(restored, 'Composition').value, 'New composition')
    assert.equal(widget(restored, 'Output duration (ms)').value, 5000)
    assert.equal(widget(restored, 'Render profile').value, 'Animate frame')
    assert.equal(widget(restored, 'Sampler').value, 'Sampler')
  }
})

test('actual panel restores historical workflows to frame continuation', () => {
  const { Node } = fixture()
  const values = state.CURRENT_WIDGET_ORDER.map(name => name)
  for (const saved of [
    { widgets_values: values },
    { widgets_values: values, properties: { duet_story_fields: Object.fromEntries(state.CURRENT_WIDGET_ORDER.map(name => [name, name])) } },
  ]) {
    const restored = new Node()
    widget(restored, 'Composition').value = 'New composition'
    restored.onConfigure(saved)
    assert.equal(widget(restored, 'Composition').value, 'Continue frame')
    assert.equal(widget(restored, 'Output duration (ms)').value, 0)
    assert.equal(widget(restored, 'Render profile').value, 'Reference shot')
    assert.equal(widget(restored, 'Memory actions').value, 'Memory actions')
  }
})

test('Add Next Shot preserves composition and both continuity wires', () => {
  const { Node, context, next } = fixture()
  const original = new Node()
  widget(original, 'Composition').value = 'New composition'
  widget(original, 'Output duration (ms)').value = 5000
  widget(original, 'Render profile').value = 'Animate frame'
  context.original = original
  vm.runInContext('createNextShot(original)', context)
  assert.equal(widget(next(), 'Composition').value, 'New composition')
  assert.equal(widget(next(), 'Output duration (ms)').value, 5000)
  assert.equal(widget(next(), 'Render profile').value, 'Animate frame')
  assert.equal(widget(next(), 'Create').value, 'Next Shot')
  assert.equal(original.links.length, 2)
  assert.deepEqual(original.links.map(([output, , input]) => [output, input]), [[2, 0], [1, 1]])
})

// The original production workflow stored the image upload last and included
// the DOM editor's marker. Its prompt must never become a mode or duration.
test('actual panel migrates original world-last production workflows', () => {
  const { Node } = fixture()
  const legacy = ['Start Story', '[]', '@Mira takes the key.', '15 seconds', 73001, '', 'Prompt mentions only', 'workshop.png']
  for (const values of [legacy, [...legacy, 'duet-story']]) {
    const restored = new Node()
    restored.onConfigure({ widgets_values: values })
    assert.equal(widget(restored, 'Reference context').value, 'Native')
    assert.equal(widget(restored, 'What happens next?').value, '@Mira takes the key.')
    assert.equal(widget(restored, 'World / starting frame').value, 'workshop.png')
    assert.equal(widget(restored, 'Shot length').value, '15 seconds')
    assert.equal(widget(restored, 'Variation').value, 73001)
    assert.equal(widget(restored, 'Sampler').value, 'Native res_multistep')
    assert.equal(widget(restored, 'Memory actions').value, '[]')
    assert.equal(widget(restored, 'Composition').value, 'Continue frame')
    assert.equal(widget(restored, 'Output duration (ms)').value, 0)
    assert.equal(widget(restored, 'Render profile').value, 'Reference shot')
    const saved = {}
    restored.onSerialize(saved)
    const reopened = new Node()
    reopened.onConfigure(saved)
    assert.equal(widget(reopened, 'What happens next?').value, '@Mira takes the key.')
    assert.equal(widget(reopened, 'World / starting frame').value, 'workshop.png')
  }
})

test('canon migration distinguishes world-last and world-first ten-field layouts', () => {
  const prompt = '@Mira keeps the key.'
  const common = ['Start Story', '[]']
  const controls = ['10 seconds', 731000, '', 'Automatic', 'Native res_multistep', '[]']
  const expected = [...common, 'Native', 'workshop.png', prompt, ...controls]
  assert.deepEqual(state.migrateSerializedStoryWidgets('DuetStoryCanon', [...common, prompt, ...controls, 'workshop.png']), expected)
  assert.deepEqual(state.migrateSerializedStoryWidgets('DuetStoryCanon', [...common, 'workshop.png', prompt, ...controls]), expected)
})

for (const sampler of ['Turbo 4-step', 'NVFP4 Exact', 'NVFP4 Balanced', 'NVFP4 Ultra Fast', 'NVFP4 Turbo 4-step']) {
test(`${sampler} selection survives save/load and Add Next Shot`, () => {
  const { Node, context, next } = fixture()
  const original = new Node()
  widget(original, 'Sampler').value = sampler
  const info = {}
  original.onSerialize(info)
  const restored = new Node()
  restored.onConfigure(info)
  assert.equal(widget(restored, 'Sampler').value, sampler)
  context.original = restored
  vm.runInContext('createNextShot(original)', context)
  assert.equal(widget(next(), 'Sampler').value, sampler)
})
}


test('prompt format survives save, reload and next-shot creation without changing legacy defaults', () => {
  const {Node,context,next} = fixture()
  const original = new Node()
  widget(original,'Prompt format').value='Structured reference (experimental)'
  const info = {}; original.onSerialize(info)
  for (const saved of [info,{widgets_values:info.widgets_values}]) {
    const restored = new Node(); restored.onConfigure(saved)
    assert.equal(widget(restored,'Prompt format').value,'Structured reference (experimental)')
  }
  context.original=original; vm.runInContext('createNextShot(original)',context)
  assert.equal(widget(next(),'Prompt format').value,'Structured reference (experimental)')
  const old = new Node(); old.onConfigure({widgets_values:info.widgets_values.slice(0,12)})
  assert.equal(widget(old,'Prompt format').value,'Current')
})


test('new nodes use automatic H3 prompting while legacy configuration restores Current', () => {
  const {Node} = fixture()
  const node = new Node(); node.onNodeCreated()
  assert.equal(widget(node,'Prompt format').value,'H3 automatic v1')
  node.onConfigure({widgets_values:state.CURRENT_WIDGET_ORDER.map(name => widget(node,name).value)})
  assert.equal(widget(node,'Prompt format').value,'Current')
})

test('saving and reopening a canvas preserves scene presence and selected state evidence', () => {
  const { Node } = fixture()
  const original = new Node()
  const fields = {'Scene entities':'["Ada"]', 'Shot state evidence':'{"Ada":"'+'a'.repeat(64)+'"}'}
  for (const [name,value] of Object.entries(fields)) original.widgets.push({name,value})
  const saved = {}
  original.onSerialize(saved)
  // Historical positional fields keep their layout; newer continuity data is named.
  assert.equal(saved.widgets_values.length,15)
  const reopened = new Node()
  for (const name of Object.keys(fields)) reopened.widgets.push({name,value:''})
  reopened.onConfigure(saved)
  const savedAgain = {}
  reopened.onSerialize(savedAgain)
  for (const [name,value] of Object.entries(fields)) {
    assert.equal(widget(reopened,name).value,value)
    assert.equal(savedAgain.properties.duet_story_fields[name],value)
  }
})
