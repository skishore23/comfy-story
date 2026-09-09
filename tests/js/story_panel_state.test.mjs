import assert from 'node:assert/strict'
import test from 'node:test'

import {
  encodeMemoryCommands,
  normalizeInspectorSummary,
  stageMemoryCommand,
  pendingMemoryCommands,
  setPendingMemoryCommands,
  recordCompletedStory,
} from '../../integrations/comfy_story/web/story_state.mjs'

const digest = 'a'.repeat(64)

test('staged action remains bound to the displayed parent revision', () => {
  const commands = stageMemoryCommand([], {
    action: 'use_in_this_shot',
    parent_revision_sha256: digest,
    target_id: 'key-bent',
  })

  assert.equal(JSON.parse(encodeMemoryCommands(commands))[0].target_id, 'key-bent')
  assert.equal(JSON.parse(encodeMemoryCommands(commands))[0].parent_revision_sha256, digest)
})

test('staged commands cannot mix authenticated parent revisions', () => {
  const commands = stageMemoryCommand([], {
    action: 'keep', parent_revision_sha256: digest, target_id: 'key-bent',
  })

  assert.throws(() => stageMemoryCommand(commands, {
    action: 'forget', parent_revision_sha256: 'b'.repeat(64), target_id: 'bad-take',
  }), /parent revision/)
})

test('invalid inspector summaries fail closed', () => {
  assert.throws(() => normalizeInspectorSummary({ revision_sha256: '../escape' }), /fields|digest/)
})

test('inspector summary retains customer-facing recall and canon sections', () => {
  const summary = normalizeInspectorSummary({
    canon: [{
      entity_id: 'brasskey', presence: 'off_screen', reference_name: 'BrassKey',
      state_note: 'bent', supporting_evidence_ids: ['key-bent'],
    }],
    important_moments: [{
      asset_sha256: 'b'.repeat(64), entity_ids: ['brasskey'], evidence_id: 'key-bent',
      frame_index: 7, kind: 'change', remembered: true, retained: false,
      salience_q: 900000, selected: true, shot_index: 6,
    }],
    owner_node_id: '42',
    pending: [],
    revision_sha256: digest,
    shot_count: 7,
    used_for_this_shot: [{ asset_sha256: 'c'.repeat(64), role: 'evidence-brasskey' }],
  })

  assert.equal(summary.canon[0].state_note, 'bent')
  assert.equal(summary.important_moments[0].selected, true)
  assert.equal(summary.used_for_this_shot[0].role, 'evidence-brasskey')
  assert.deepEqual(normalizeInspectorSummary([summary]), summary)
  assert.equal(normalizeInspectorSummary({ ...summary, video_sha256: digest }).video_sha256, digest)
  assert.throws(() => normalizeInspectorSummary({ ...summary, video_sha256: '../escape' }), /digest/)
  assert.throws(() => normalizeInspectorSummary([summary, summary]), /object/)
  const node = { widgets: [{ name: 'Memory actions', value: 'unchanged generating inputs' }], properties: {} }
  recordCompletedStory(node, summary)
  setPendingMemoryCommands(node, [{ action: 'keep', parent_revision_sha256: digest, target_id: 'key-bent' }])
  assert.equal(pendingMemoryCommands(node)[0].target_id, 'key-bent')
  assert.equal(node.widgets[0].value, 'unchanged generating inputs')
  const reopened = { properties: JSON.parse(JSON.stringify(node.properties)) }
  recordCompletedStory(reopened, reopened.properties.comfy_story_inspector)
  assert.equal(pendingMemoryCommands(reopened).length, 1)
  recordCompletedStory(reopened, { ...summary, revision_sha256: 'd'.repeat(64) })
  assert.deepEqual(pendingMemoryCommands(reopened), [])
  assert.throws(() => setPendingMemoryCommands(reopened, [{ action: 'keep', parent_revision_sha256: digest, target_id: 'key-bent' }]), /displayed completed/)
})
