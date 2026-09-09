import assert from 'node:assert/strict'
import test from 'node:test'

import {
  deriveStoryView,
  encodeLibrary,
  extractMentions,
  keepAsCurrentLook,
  nextShotWiring,
  normalizeProtectedNames,
} from '../../integrations/comfyui_duetx_continuity/web/story_state.mjs'

const library = (count = 1) => ({
  project_name: 'Compiler preview',
  references: Array.from({ length: count }, (_, index) => ({
    file: `ref-${index}.png`,
    name: `Ref${index}`,
    note: '',
    role: index === 0 ? 'Character' : 'Prop',
  })),
})

const values = (overrides = {}) => ({
  Create: 'Start Story',
  'Keep this detail': '',
  'Reference context': 'Native',
  'Story Library': encodeLibrary(library()),
  'What happens next?': '@Ref0 enters.',
  ...overrides,
})

test('extractMentions preserves first occurrence and canonical prompt order', () => {
  assert.deepEqual(extractMentions('@Maya sees @Chest, then @maya leaves.'), ['Maya', 'Chest'])
})

test('compiled preview requires at least two selected visual sources', () => {
  const view = deriveStoryView(values({
    'Reference context': 'Compiled preview',
    'What happens next?': 'An empty establishing shot.',
  }), { story: false, frame: false, motion: false })

  assert.equal(view.ready, false)
  assert.match(view.message, /two visual references/)
})

test('native MiniMax roster accepts eight named references', () => {
  const prompt = Array.from({ length: 8 }, (_, index) => `@Ref${index}`).join(' meets ')
  const view = deriveStoryView(values({
    'Story Library': encodeLibrary(library(8)),
    'What happens next?': prompt,
  }), { story: false, frame: false, motion: false })

  assert.equal(view.ready, true)
  assert.equal(view.active.length, 8)
})

test('motion reference occupies one of the nine visual roster slots', () => {
  const prompt = Array.from({ length: 8 }, (_, index) => `@Ref${index}`).join(' ')
  const view = deriveStoryView(values({
    'Story Library': encodeLibrary(library(8)),
    'What happens next?': prompt,
  }), { story: false, frame: false, motion: true })

  assert.equal(view.ready, false)
  assert.match(view.message, /at most 7 named references/)
})

test('protected names must exist in the selected library', () => {
  assert.throws(
    () => normalizeProtectedNames('@Maya, @Unknown', ['Maya']),
    /@Unknown is not in the Story Library/,
  )
})

test('protected names must also be active in this shot and are capped at two', () => {
  assert.throws(
    () => normalizeProtectedNames('@Chest', ['Maya', 'Chest'], ['Maya']),
    /@Chest must be selected/,
  )
  assert.throws(
    () => normalizeProtectedNames('@A,@B,@C', ['A', 'B', 'C']),
    /at most two/,
  )
})

test('deriveStoryView returns exact-detail chips in library spelling', () => {
  const view = deriveStoryView(values({
    'Keep this detail': '@ref0',
    'Reference context': 'Compiled preview',
  }), { story: false, frame: false, motion: false })

  assert.equal(view.ready, true)
  assert.deepEqual(view.protected, ['@Ref0'])
  assert.match(view.message, /2 visual sources/)
})

test('nextShotWiring carries both story state and last frame', () => {
  const calls = []
  const source = { connect: (...args) => calls.push(args) }
  const target = {}

  const result = nextShotWiring(source, target)

  assert.deepEqual(calls, [[2, target, 0], [1, target, 1]])
  assert.deepEqual(result, [
    { from: 2, to: 0, type: 'DUET_STORY' },
    { from: 1, to: 1, type: 'IMAGE' },
  ])
})

test('keepAsCurrentLook makes confirmed visual state durable in one action', () => {
  const parent = 'a'.repeat(64)

  const result = keepAsCurrentLook([], {
    parentRevisionSha256: parent,
    entityId: 'redparcel',
    evidenceId: 'shot-000-opening-000-c1279d07a953',
    stateNote: 'blue ribbon and yellow duck charm',
    presence: 'off_screen',
  })

  assert.deepEqual(result, [
    {
      action: 'keep',
      parent_revision_sha256: parent,
      presence: null,
      state_note: '',
      supporting_evidence_ids: [],
      target_id: 'shot-000-opening-000-c1279d07a953',
    },
    {
      action: 'update_canon',
      parent_revision_sha256: parent,
      presence: 'off_screen',
      state_note: 'blue ribbon and yellow duck charm',
      supporting_evidence_ids: ['shot-000-opening-000-c1279d07a953'],
      target_id: 'redparcel',
    },
  ])
})
