import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import * as state from '../../integrations/comfy_story/web/story_state.mjs'

// Execute the actual panel lifecycle; helper-only tests miss serialization hooks.
const source = readFileSync(new URL('../../integrations/comfy_story/web/story_panel.js', import.meta.url), 'utf8').replace(/^import .*\n/gm, '')
const names = [...state.CURRENT_WIDGET_ORDER, 'Composition', 'Output duration (ms)', 'Render profile', 'Prompt format']
function fixture(document) {
  let extension
  let next
  const context = vm.createContext({
    ...state,
    document,
    app: { registerExtension: value => { extension = value }, graph: { add: value => { next = value } } },
    api: { addEventListener() {} },
    queueMicrotask: fn => fn(),
    console,
  })
  vm.runInContext(source, context)
  class Node {
    constructor() {
      this.widgets = names.map(name => ({ name, value: name === 'Shot length' ? '5 seconds' : name === 'Variation' ? 7 : name === 'Composition' ? 'Continue frame' : name === 'Prompt format' ? 'Current' : name === 'Output duration (ms)' ? 0 : name }))
      this.properties = {}
      this.links = []
    }
    connect(...args) { this.links.push(args) }
  }
  extension.beforeRegisterNodeDef(Node, { name: 'ComfyStory' })
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
  assert.equal(info.widgets_values.length, names.length)
  assert.equal(info.properties.comfy_story_fields.Composition, 'New composition')
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
  const defaults = new Node()
  const values = state.CURRENT_WIDGET_ORDER.map(name => widget(defaults,name).value)
  for (const saved of [
    { widgets_values: values },
    { widgets_values: values, properties: { comfy_story_fields: Object.fromEntries(state.CURRENT_WIDGET_ORDER.map(name => [name, name])) } },
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
test('unsupported world-last layouts fail before changing visible widgets', () => {
  const { Node } = fixture()
  const old = ['Start Story', '[]', '@Mira takes the key.', '15 seconds', 73001, '', 'Prompt mentions only', 'workshop.png']
  for (const values of [old, [...old, 'comfy-story']]) {
    const restored = new Node()
    const before = restored.widgets.map(widget => widget.value)
    assert.throws(() => restored.onConfigure({widgets_values:values}), /Unsupported.*layout/)
    assert.deepEqual(restored.widgets.map(widget => widget.value), before)
  }
})

test('restoration validates field positions and accepts only the current node identity', () => {
  const common = ['Start Story', '[]']
  const controls = ['10 seconds', 731000, '', 'Automatic', 'Native res_multistep', '[]']
  const current = [...common, 'workshop.png', '@Mira keeps the key.', ...controls]
  assert.deepEqual(state.restoreSerializedStoryWidgets(current), current)
  assert.throws(() => state.restoreSerializedStoryWidgets([...common, '@Mira keeps the key.', ...controls, 'workshop.png']), /Unsupported.*layout/)
  assert.equal(state.isComfyStoryNode({type:'ComfyStoryCanon'}), false)
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
  assert.equal(saved.widgets_values.length,names.length)
  const reopened = new Node()
  for (const name of Object.keys(fields)) reopened.widgets.push({name,value:''})
  reopened.onConfigure(saved)
  const savedAgain = {}
  reopened.onSerialize(savedAgain)
  for (const [name,value] of Object.entries(fields)) {
    assert.equal(widget(reopened,name).value,value)
    assert.equal(savedAgain.properties.comfy_story_fields[name],value)
  }
})

// Minimal DOM implements real selector lookup over parsed panel markup, so missing
// controls fail during mounting instead of being hidden by a no-op widget fixture.
class PanelElement {
  constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.dataset = {}; this.style = {}; this.listeners = {}; this.value = '' }
  append(...items) { for (const item of items) { item.parent = this; this.children.push(item) } }
  appendChild(item) { this.append(item); return item }
  replaceChildren(...items) { this.children = []; this.append(...items) }
  setAttribute(name, value) { this.attrs[name] = value; if (name === 'class') this.className = value }
  addEventListener(name, fn) { (this.listeners[name] ??= []).push(fn) }
  async fire(name) { for (const fn of this.listeners[name] ?? []) await fn({}) }
  before(...items) { this.insertSiblings(0, items) }
  after(...items) { this.insertSiblings(1, items) }
  insertSiblings(offset, items) { for (const item of items) item.parent = this.parent; this.parent.children.splice(this.parent.children.indexOf(this) + offset, 0, ...items) }
  all() { return this.children.flatMap(child => [child, ...child.all()]) }
  querySelectorAll(selector) {
    return this.all().filter(item => selector.split(',').some(part => {
      const key = part.trim()
      return key.startsWith('.') ? (item.className ?? '').split(' ').includes(key.slice(1)) : item.tag === key
    }))
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null }
  set innerHTML(markup) {
    this.children = []
    const stack = [this]
    for (const match of markup.matchAll(/<\/?([a-z]+)\b[^>]*>/g)) {
      if (match[0].startsWith('</')) { stack.pop(); continue }
      const element = new PanelElement(match[1])
      for (const attr of match[0].matchAll(/([\w-]+)="([^"]*)"/g)) element.setAttribute(attr[1], attr[2])
      stack.at(-1).append(element)
      if (!['input', 'img', 'br', 'hr'].includes(element.tag)) stack.push(element)
    }
  }
}

test('actual panel mounts, edits a shot, queues generation and stages memory', async () => {
  const document = {head: new PanelElement('head'), createElement: tag => new PanelElement(tag), getElementById: () => null}
  const {Node, context, next} = fixture(document)
  Node.prototype.addDOMWidget = function (name, type, root) { this.panel = root }
  const node = new Node()
  widget(node, 'Create').value = 'Start Story'
  widget(node, 'Story Library').value = state.encodeLibrary({project_name:'A story', references:[{name:'Boat', role:'Prop', file:'boat.png', note:''}]})
  widget(node, 'What happens next?').value = 'A paper boat drifts.'
  assert.doesNotThrow(() => node.onNodeCreated())
  assert.ok(node.panel)
  const controls = node.panel.all().filter(item => item.attrs['aria-label'])
  assert.ok(controls.some(item => item.attrs['aria-label'] === 'What happens next?'))
  assert.ok(!controls.some(item => item.attrs['aria-label'] === 'Reference context'))
  const prompt = controls.find(item => item.attrs['aria-label'] === 'What happens next?')
  prompt.value = 'The boat reaches the bank.'
  await prompt.fire('input')
  assert.equal(widget(node, 'What happens next?').value, prompt.value)
  let queued = 0
  context.app.queuePrompt = async () => { queued++ }
  const run = node.panel.querySelector('.comfy-story-primary')
  assert.equal(run.disabled, false)
  await run.fire('click')
  assert.equal(queued, 1)
  node.__comfyInspector = {revision_sha256:'a'.repeat(64), shot_count:1, used_for_this_shot:[], canon:[], pending:[], important_moments:[
    {evidence_id:'boat-bank', entity_ids:[], kind:'scene', shot_index:0, frame_index:0, asset_sha256:'b'.repeat(64), retained:false},
  ]}
  context.api.apiURL = path => path
  node.__comfyRefresh()
  const keep = node.panel.all().find(item => item.tag === 'button' && item.textContent === 'Keep this moment')
  assert.ok(keep)
  await keep.fire('click')
  assert.equal(state.pendingMemoryCommands(node)[0].action, 'keep')
  assert.equal(state.pendingMemoryCommands(node)[0].target_id, 'boat-bank')
  await node.panel.querySelector('.comfy-story-next').fire('click')
  assert.equal(widget(next(), 'Create').value, 'Next Shot')
  assert.equal(JSON.parse(widget(next(), 'Memory actions').value)[0].target_id, 'boat-bank')
  assert.equal(node.links.length, 2)
})
