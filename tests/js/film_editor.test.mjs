import test from 'node:test'
import assert from 'node:assert/strict'
import { validateEditorSeeds, parseShotSeconds, filmNodeDuration, parseEndingCounts, filmRunStage, filmImagePreviewPath, filmRenderApproach, setFilmRenderApproach } from '../../integrations/comfy_story/web/film_editor.mjs'

test('precise cut lengths retain exact frames and reject silent rounding', () => {
  for (const seconds of ['0.125', '2.375', '2.5', '3', '9', '15']) {
    assert.equal(parseShotSeconds(seconds), Number(seconds) * 1000)
  }
  for (const seconds of ['', ' ', '0', '-1', '15.125', '2.1', '1.333333333', 'NaN', 'Infinity']) {
    assert.throws(() => parseShotSeconds(seconds), /exact video frames/)
  }
})

test('a node cut overrides its render block without accepting impossible durations', () => {
  assert.equal(filmNodeDuration({'Shot length':'5 seconds','Output duration (ms)':2375}),2375)
  assert.equal(filmNodeDuration({'Shot length':'5 seconds','Output duration (ms)':0}),5000)
  assert.equal(filmNodeDuration({'Shot length':'15 seconds'}),15000)
  assert.equal(filmNodeDuration({}),10000)
  for (const cut of [-1, 5125, 2100, NaN, Infinity, '2375', 2375.5]) {
    assert.throws(()=>filmNodeDuration({'Shot length':'5 seconds','Output duration (ms)':cut}))
  }
})

test('run stages distinguish rendering and reviewing without guessing legacy progress', () => {
  assert.equal(filmRunStage({phase:'rendering_shot'}), 'Rendering the current shot')
  assert.equal(filmRunStage({phase:'checking_shot'}), 'Reviewing the current shot')
  assert.equal(filmRunStage({phase:'checking_film'}), 'Reviewing the complete film')
  assert.equal(filmRunStage({}), null)
  assert.equal(filmRunStage({phase:'unknown'}), null)
  assert.equal(filmRunStage({phase:'constructor'}), null)
})

test('editor refuses unsafe seeds instead of saving rounded values', () => {
  for (const seed of [2**53, -1, NaN, Infinity, 1.5, '42']) {
    const inputs = {shots_by_id: {one: {variation: seed}}}
    assert.throws(() => validateEditorSeeds(inputs), /represented exactly/)
    assert.equal(inputs.shots_by_id.one.variation, seed)
  }
  for (const seed of [0, 42, Number.MAX_SAFE_INTEGER]) {
    assert.doesNotThrow(() => validateEditorSeeds({shots_by_id:{one:{variation:seed}}}))
  }
})

import { openFilmEditor } from '../../integrations/comfy_story/web/film_editor.mjs'
import { filmStoryboardRows, filmCoverageText } from '../../integrations/comfy_story/web/film_editor.mjs'

test('story overview exposes causality conflicts and timing without editing the plan', () => {
  const plan = {initial_facts:[{key:'Gate.position',value:'closed'}],shots:[
    {shot_id:'open',duration_ms:2375,purpose:'Open',action:'The gate opens.',effects:[{key:'Gate.position',value:'open'}]},
    {shot_id:'pass',duration_ms:5000,purpose:'Pass',action:'A cart passes.',requires:[{key:'Gate.position',value:'open'}]},
  ]}
  const before = structuredClone(plan)
  const rows = filmStoryboardRows(plan)
  assert.deepEqual(rows.map(x=>[x.start_ms,x.end_ms]),[[0,2375],[2375,7375]])
  assert.deepEqual(rows[0].changes,['Gate.position: closed → open'])
  assert.deepEqual(rows[1].conflicts,[])
  assert.deepEqual(plan,before)
  const reordered = filmStoryboardRows({...plan,shots:[plan.shots[1],plan.shots[0]]})
  assert.deepEqual(reordered[0].conflicts,['Gate.position: needs open; earlier plan leaves closed'])
  assert.equal(reordered[0].action,'A cart passes.')
  assert.deepEqual(filmStoryboardRows({...plan,initial_facts:[],shots:[plan.shots[1]]})[0].conflicts,
    ['Gate.position: needs open; earlier plan leaves undeclared'])
})

test('run coverage separates a rejected preview from selected duration and creator approval', () => {
  const run = {status:'needs_attention',preview:{duration_ms:15000},coverage:{target_duration_ms:60000,
    planned_shots:7,selected_duration_ms:10000,selected_shots:1}}
  assert.equal(filmCoverageText(run),'Incomplete film. 10 of 60 seconds selected by machine checks (1/7 shots).')
  assert.equal(filmCoverageText({status:'needs_attention'}),null)
  assert.equal(filmCoverageText({...run,status:'running'}),'10 of 60 seconds selected by machine checks (1/7 shots).')
  assert.equal(filmCoverageText({...run,status:'ready_for_review',coverage:{...run.coverage,selected_shots:7,selected_duration_ms:60000}}),
    '60 of 60 seconds selected by machine checks (7/7 shots).')
})

class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.listeners = {}; this.attrs = {}; this.isConnected = false; this.value = ''; this.validityMessage = '' }
  append(child) { child.parent = this; child.isConnected = true; this.children.push(child) }
  replaceChildren() { this.children = [] }
  getAttribute(name) { return this.attrs[name] ?? null }
  contains(child) { return this === child || this.all().includes(child) }
  setAttribute(name,value) { this.attrs[name] = value }
  removeAttribute(name) { delete this.attrs[name]; delete this[name] }
  addEventListener(name,callback) { (this.listeners[name] ??= []).push(callback) }
  async fire(name) { for (const callback of this.listeners[name] ?? []) await callback({}) }
  setCustomValidity(value) { this.validityMessage = value }
  get validationMessage() { return this.validityMessage }
  get parentElement() { return this.parent }
  get tagName() { return this.tag.toUpperCase() }
  reportValidity() {}
  showModal() {}
  focus() {}
  close() { return this.fire('close') }
  remove() { this.parent.children = this.parent.children.filter(x=>x!==this); this.isConnected = false }
  querySelector() { return this.all().find(x=>x.validityMessage) || null }
  all() { return this.children.flatMap(x=>[x,...x.all()]) }
  get style() { return this._style ??= {} }
}

test('shot disclosures preserve editing state by identity without changing recipes or seeds', async () => {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  let stored = {revision:'a'.repeat(64),plan:{project_id:'disclosures',title:'Two events',target_duration_ms:10000,initial_facts:[],shots:[
    {shot_id:'one',duration_ms:5000,purpose:'A door opens',action:'The door opens.',present:[]},
    {shot_id:'two',duration_ms:5000,purpose:'A leaf moves',action:'A leaf moves.',present:[],camera_policy:'Single continuous shot'},
  ]},inputs:{library:{project_name:'Two events',references:[]},shots_by_id:{one:{variation:42,world:'door.png'},two:{variation:43,world:'leaf.png'}}}}
  const original = structuredClone(stored)
  let saves = 0
  const api = {apiURL:path=>path,fetchApi:async(path,options)=>{
    if (options.method === 'PUT') { stored={...JSON.parse(options.body),revision:'b'.repeat(64)}; saves++ }
    return {ok:true,json:async()=>path==='/comfy/story/films' ? {projects:[]} : {recipe:structuredClone(stored),runs:[],verification_configured:false}}
  }}
  const cards = () => body.all().filter(x=>x.tag==='details' && x.children[0]?.tag==='summary' && /^\d+\./.test(x.children[0].textContent))
  const button = label => body.all().find(x=>x.tag==='button' && x.textContent===label)
  await openFilmEditor(api,{properties:{comfy_film_project_id:'disclosures'}},{},document)
  const overview = () => body.all().find(x=>x.attrs['aria-label']==='Story at a glance')
  assert.ok(overview().all().some(x=>x.textContent==='0–5s'))
  assert.ok(overview().all().some(x=>x.textContent==='5–10s'))
  assert.deepEqual(cards().map(x=>x.open),[true,false])
  assert.equal(cards()[1].children[0].textContent,'2. A leaf moves · 5 seconds')
  cards()[0].open=false; await cards()[0].fire('toggle')
  cards()[1].open=true; await cards()[1].fire('toggle')
  assert.equal(saves,0)
  assert.deepEqual(stored,original)
  const second = cards()[1]
  const timing = second.all().find(x=>x.attrs['aria-label']==='Seconds')
  timing.value='2.375'; await timing.fire('input'); await timing.fire('change')
  assert.ok(overview().all().some(x=>x.textContent==='5–7.375s'))
  assert.deepEqual(cards().map(x=>x.open),[false,true])
  assert.equal(cards()[1].children[0].textContent,'2. A leaf moves · 2.375 seconds')
  await cards()[1].all().find(x=>x.tag==='button' && x.textContent==='Move earlier').fire('click')
  assert.deepEqual(cards().map(x=>x.open),[true,false], 'Disclosure follows the shot, not its position')
  assert.equal(cards()[0].children[0].textContent,'1. A leaf moves · 2.375 seconds')
  await button('Save plan').fire('click')
  assert.equal(saves,1)
  assert.equal(stored.plan.target_duration_ms,7375)
  assert.deepEqual(stored.inputs,original.inputs)
  assert.deepEqual(Object.keys(stored.plan.shots[0]).sort(),Object.keys(original.plan.shots[1]).sort())
  assert.equal(stored.plan.shots[0].camera_policy,'Single continuous shot')
  assert.equal(stored.plan.shots[0].direction_version,undefined, 'Saving a legacy shot must not migrate its prompt')
  assert.deepEqual(cards().map(x=>x.open),[true,false], 'Saving preserves session disclosure state')
  await button('Close').fire('click')
})

test('advanced controls stay optional, preserve saved settings, and reveal invalid fields', async () => {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  let stored = {revision:'a'.repeat(64),plan:{project_id:'simple',title:'A paper boat',target_duration_ms:5000,initial_facts:[],shots:[
    {shot_id:'one',duration_ms:5000,purpose:'Drift',action:'A paper boat drifts.',present:[],camera_policy:'Locked frame',direction_version:1},
  ]},inputs:{library:{project_name:'A paper boat',references:[]},recall_selected_state:true,shots_by_id:{one:{variation:42,world:'boat.png',ending_frame:'end.png',sampler:'Turbo 8-step',render_profile:'Animate frame'}}}}
  const original = structuredClone(stored)
  let saves = 0
  const api = {apiURL:path=>path,fetchApi:async(path,options)=>{
    if (options.method === 'PUT') { stored={...JSON.parse(options.body),revision:'b'.repeat(64)}; saves++ }
    return {ok:true,json:async()=>path==='/comfy/story/films' ? {projects:[]} : {recipe:structuredClone(stored),runs:[],verification_configured:false}}
  }}
  const control = label => body.all().find(x=>x.attrs['aria-label']===label)
  const button = label => body.all().find(x=>x.tag==='button' && x.textContent===label)
  const details = label => body.all().find(x=>x.tag==='details' && x.children[0]?.textContent===label)
  const node = {properties:{comfy_film_project_id:'simple'}}
  await openFilmEditor(api,node,{},document)
  for (const label of ['Advanced shot settings','Advanced film settings']) assert.equal(details(label).open,false)
  const advanced = details('Advanced shot settings')
  for (const label of ['Locked seed','Sampler','Required starting facts','Present reference names']) {
    assert.ok(advanced.all().includes(control(label)), label)
  }
  for (const label of ['Purpose','Visible action','Seconds','Render approach','Camera policy','Ending image (optional)','Starting image (blank uses previous frame)']) {
    assert.ok(!advanced.all().includes(control(label)), label)
  }
  assert.ok(details('Advanced film settings').all().includes(control('Appearance references')))
  assert.ok(details('Project tools and run settings').all().includes(control('Maximum attempts per shot')))
  assert.equal(button('Resume selected run').hidden,true)
  assert.equal(button('Pause after current shot').hidden,true)
  advanced.open=true; await advanced.fire('toggle')
  await button('Save plan').fire('click')
  assert.equal(saves,1)
  assert.deepEqual(stored.plan,original.plan)
  assert.deepEqual(stored.inputs,original.inputs)
  assert.equal(details('Advanced shot settings').open,true)
  control('Locked seed').value='not a seed'; await control('Locked seed').fire('input')
  details('Advanced shot settings').open=false
  const card = body.all().find(x=>x.className==='comfy-film-shot'); card.open=false
  await button('Save plan').fire('click')
  assert.equal(saves,1)
  assert.equal(details('Advanced shot settings').open,true)
  assert.equal(card.open,true)
  assert.equal(stored.inputs.shots_by_id.one.variation,42)
  await button('Close').fire('click')
  await button('Discard draft and close').fire('click')
  await openFilmEditor(api,node,{},document)
  assert.equal(details('Advanced shot settings').open,false)
  assert.equal(control('Locked seed').value,'42')
  assert.equal(control('Sampler').value,'Turbo 8-step')
  await button('Close').fire('click')
})

test('editor reopens precise timings and preserves seed while editing a cut', async () => {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  let stored = {revision:'a'.repeat(64),plan:{project_id:'cut',title:'A short action',target_duration_ms:2500,initial_facts:[],shots:[{shot_id:'one',duration_ms:2500,purpose:'Act',action:'A leaf turns.',present:[]}]},inputs:{library:{references:[]},shots_by_id:{one:{variation:42,world:'leaf.png'}}}}
  let saves = 0
  const api = {apiURL:path=>path,fetchApi:async(path,options)=>{
    if (options.method === 'PUT') { stored={...JSON.parse(options.body),revision:'b'.repeat(64)}; saves++ }
    return {ok:true,json:async()=>path==='/comfy/story/films' ? {projects:[]} : {recipe:structuredClone(stored),runs:[],verification_configured:false}}
  }}
  const node={properties:{comfy_film_project_id:'cut'}}
  const control=label=>body.all().find(x=>x.attrs['aria-label']===label)
  const button=label=>body.all().find(x=>x.tag==='button' && x.textContent===label)
  await openFilmEditor(api,node,{},document)
  assert.equal(control('Seconds').tag,'input')
  assert.equal(control('Seconds').value,'2.5')
  const timing=control('Seconds')
  timing.value='2.375'; await timing.fire('input')
  assert.equal(control('Seconds'),timing, 'Typing must not replace the focused input')
  await timing.fire('change')
  await button('Save plan').fire('click')
  assert.equal(stored.plan.shots[0].duration_ms,2375)
  assert.equal(stored.plan.target_duration_ms,2375)
  assert.equal(stored.inputs.shots_by_id.one.variation,42)
  const before=saves
  control('Seconds').value='2.1'; await control('Seconds').fire('input'); await control('Seconds').fire('change')
  assert.match(control('Seconds').validationMessage,/exact video frames/)
  await button('Save plan').fire('click')
  assert.equal(saves,before)
  await button('Close').fire('click')
  await button('Discard draft and close').fire('click')
  await openFilmEditor(api,node,{},document)
  assert.equal(control('Seconds').value,'2.375')
})

test('opening Story from a trimmed Comfy node saves its actual interval and seed', async () => {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  let stored
  const api = {apiURL:path=>path,fetchApi:async(path,options)=>{
    if (options.method==='POST') stored={...JSON.parse(options.body),revision:'c'.repeat(64)}
    return {ok:true,json:async()=>path==='/comfy/story/films' && options.method==='GET' ? {projects:[]} : {recipe:structuredClone(stored),runs:[],verification_configured:false}}
  }}
  await openFilmEditor(api,{properties:{}},{'Story Library':JSON.stringify({project_name:'Precise node',references:[]}), 'What happens next?':'A leaf turns.', 'World / starting frame':'leaf.png', 'Shot length':'5 seconds','Output duration (ms)':2375,Variation:73},document)
  await body.all().find(x=>x.tag==='button' && x.textContent==='Save plan').fire('click')
  assert.equal(stored.plan.target_duration_ms,2375)
  assert.equal(stored.plan.shots[0].duration_ms,2375)
  assert.equal(stored.inputs.shots_by_id['shot-1'].variation,73)
  assert.equal(body.all().find(x=>x.attrs['aria-label']==='Seconds').value,'2.375')
})

for (const renderProfile of ['Animate frame', 'Reference shot']) for (const cameraPolicy of ['Locked frame', 'Single continuous shot']) test(`actual editor preserves ${renderProfile}, ${cameraPolicy} and the seed and blocks invalid seeds`, async () => {
  const body = new Element('body')
  const document = { body, createElement:tag=>new Element(tag), getElementById:id=>body.all().find(x=>x.id===id) }
  let stored, saves = 0
  const api = { apiURL:path=>path, fetchApi:async (path,options) => {
    let data
    if (options.method==='POST' || options.method==='PUT') {
      const request = JSON.parse(options.body)
      stored = {...request, revision:'a'.repeat(64)}; saves++; data={recipe:stored}
    } else if (path==='/comfy/story/films') data={projects:[]}
    else data={recipe:stored,runs:[],verification_configured:false}
    return {ok:true,json:async()=>structuredClone(data)}
  }}
  const node = {properties:{}}
  const values = {'Story Library':JSON.stringify({project_name:'Test',references:[]}), 'What happens next?':'A leaf falls.', 'World / starting frame':'world.png',Variation:42}
  const control = label => body.all().find(x=>x.attrs['aria-label']===label)
  const button = label => body.all().find(x=>x.tag==='button' && x.textContent===label)
  await openFilmEditor(api,node,values,document)
  control('Must remain fully visible').value='Vessel, Lantern'; await control('Must remain fully visible').fire('input')
  control('Locked seed').value='123456'; await control('Locked seed').fire('input')
  control('Ending visible counts (optional)').value='red mugs=2'; await control('Ending visible counts (optional)').fire('input')
  control('Camera policy').value=cameraPolicy; await control('Camera policy').fire('change')
  control('Frame borders').value='Preserve opening borders'; await control('Frame borders').fire('change')
  control('Ending image (optional)').value='guides/end.png'; await control('Ending image (optional)').fire('input')
  control('Render profile').value=renderProfile; await control('Render profile').fire('change')
  control('Sampler').value='Turbo 8-step'; await control('Sampler').fire('change')
  control('Appearance references').value='Selected state evidence'; await control('Appearance references').fire('change')
  await button('Save plan').fire('click')
  assert.equal(stored.inputs.shots_by_id['shot-1'].variation,123456)
  assert.equal(stored.inputs.recall_selected_state,true)
  control('Locked seed').value='123456'; await control('Locked seed').fire('input')
  await button('Add shot').fire('click')
  assert.equal(body.all().filter(x=>x.attrs['aria-label']==='Render profile').at(-1).value,renderProfile)
  assert.equal(body.all().filter(x=>x.attrs['aria-label']==='Sampler').at(-1).value,'Turbo 8-step')
  assert.deepEqual(stored.plan.shots[0].ending_counts,[{category:'red mugs',count:2}])
  assert.equal(stored.plan.shots[0].camera_policy,cameraPolicy)
  assert.equal(stored.plan.shots[0].direction_version,2)
  assert.equal(stored.plan.shots[0].border_policy,'Preserve opening borders')
  assert.equal(stored.inputs.shots_by_id['shot-1'].ending_frame,'guides/end.png')
  assert.equal(stored.inputs.shots_by_id['shot-1'].render_profile || 'Reference shot',renderProfile)
  assert.deepEqual(stored.plan.shots[0].fully_visible_throughout,['Vessel','Lantern'])
  assert.equal(saves,1)
  await button('Close').fire('click')
  await openFilmEditor(api,node,values,document)
  assert.equal(control('Must remain fully visible').value,'Vessel, Lantern')
  assert.equal(control('Locked seed').value,'123456')
  assert.equal(control('Ending visible counts (optional)').value,'red mugs=2')
  assert.equal(control('Camera policy').value,cameraPolicy)
  assert.equal(control('Frame borders').value,'Preserve opening borders')
  assert.equal(control('Ending image (optional)').value,'guides/end.png')
  assert.equal(control('Render profile').value,renderProfile)
  assert.equal(control('Sampler').value,'Turbo 8-step')
  control('Locked seed').value=String(2**53); await control('Locked seed').fire('input')
  await button('Save plan').fire('click')
  assert.equal(saves,1)
  await button('Generate saved plan').fire('click')
  assert.equal(saves,1)
  assert.equal(stored.inputs.shots_by_id['shot-1'].variation,123456)
  assert.equal(stored.inputs.recall_selected_state,true)
})


test('a new Comfy node opens an editable film before any references are configured', async () => {
  const body = new Element('body')
  const document = { body, createElement:tag=>new Element(tag), getElementById:id=>body.all().find(x=>x.id===id) }
  let stored
  const api = { fetchApi:async(path, options)=> {
    if (options.method==='POST') { stored = JSON.parse(options.body); assert.equal(stored.inputs.library.project_name,'My first film'); return {ok:true,json:async()=>({recipe:{...stored,revision:'a'.repeat(64)}})} }
    return {ok:true,json:async()=>path==='/comfy/story/films' ? {projects:[]} : {verification_configured:true}}
  }}
  await openFilmEditor(api,{properties:{}},{'Story Library':'[]','What happens next?':'','World / starting frame':'None'},document)
  assert.ok(body.all().find(x=>x.attrs['aria-label']==='Film title'))
  assert.ok(body.all().find(x=>x.tag==='button' && x.textContent==='Add reference'))
  assert.ok(body.all().find(x=>x.attrs['aria-label']==='Visible action'))
  const title = body.all().find(x=>x.attrs['aria-label']==='Film title')
  title.value = 'My first film'; await title.fire('input')
  await body.all().find(x=>x.tag==='button' && x.textContent==='Save plan').fire('click')
  assert.equal(stored.plan.title,'My first film')
})


test('an untouched new node opens a saved project and remembers it without saving a new draft', async () => {
  const body = new Element('body')
  const document = { body, createElement:tag=>new Element(tag), getElementById:id=>body.all().find(x=>x.id===id) }
  const recipe = {revision:'a'.repeat(64),plan:{project_id:'saved',title:'Saved film',shots:[],initial_facts:[]},inputs:{library:{references:[]},shots_by_id:{}}}
  const reads = []
  const api = {fetchApi:async(path,options)=> {
    assert.equal(options.method,'GET'); reads.push(path)
    return {ok:true,json:async()=>path==='/comfy/story/films' ? {projects:[{project_id:'saved',title:'Saved film'}]} : {recipe:structuredClone(recipe),runs:[],verification_configured:true}}
  }}
  const node = {properties:{}}
  const values = {'Story Library':'[]','World / starting frame':'None'}
  const control = label => body.all().find(x=>x.attrs['aria-label']===label)
  const close = () => body.all().find(x=>x.tag==='button' && x.textContent==='Close').fire('click')
  await openFilmEditor(api,node,values,document)
  control('Open film project').value='saved'; await control('Open film project').fire('change')
  assert.equal(control('Film title').value,'Saved film')
  assert.equal(node.properties.comfy_film_project_id,'saved')
  await close(); await openFilmEditor(api,node,values,document)
  assert.equal(control('Film title').value,'Saved film')
  control('Film title').value='Unsaved edit'; await control('Film title').fire('input')
  const before=reads.length
  control('Open film project').value='another'; await control('Open film project').fire('change')
  assert.equal(reads.length,before)
  assert.equal(control('Film title').value,'Unsaved edit')
  assert.equal(control('Open film project').value,'saved')
})


test('pending film launch shows progress and suppresses duplicate generate or resume requests', async () => {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  const recipe = {revision:'a'.repeat(64),plan:{project_id:'saved',title:'Saved film',shots:[],initial_facts:[]},inputs:{library:{references:[]},shots_by_id:{}}}
  let finish, posts = 0
  const pending = new Promise(resolve => { finish = resolve })
  const api = {fetchApi:async(path,options)=> {
    if (options.method === 'POST') { posts++; await pending; return {ok:false,json:async()=>({error:'Missing configured model'})} }
    return {ok:true,json:async()=>path==='/comfy/story/films' ? {projects:[]} : {recipe:structuredClone(recipe),runs:[],verification_configured:true}}
  }}
  await openFilmEditor(api,{properties:{comfy_film_project_id:'saved'}},{},document)
  const button = name => body.all().find(x=>x.tag==='button' && x.textContent===name)
  const generate = button('Generate saved plan'), resume = button('Resume selected run')
  const first = generate.fire('click')
  await Promise.resolve(); await Promise.resolve()
  assert.equal(posts,1)
  assert.equal(generate.disabled,true)
  assert.equal(resume.disabled,true)
  assert.match(body.all().find(x=>x.attrs.role==='status').textContent,/Checking model and input files/)
  assert.equal(button('New film').disabled,true)
  await button('New film').fire('click')
  assert.equal(button('Save plan').disabled,true)
  await button('Save plan').fire('click')
  await generate.fire('click'); await resume.fire('click')
  assert.equal(posts,1)
  finish(); await first
  assert.equal(generate.disabled,false)
  assert.equal(resume.disabled,false)
  assert.match(body.all().find(x=>x.attrs.role==='status').textContent,/Missing configured model/)
})


test('ending count editor accepts optional category counts and rejects ambiguous numbers', () => {
  assert.deepEqual(parseEndingCounts(''), [])
  assert.deepEqual(parseEndingCounts('red mugs=2\npeople=0'), [{category:'red mugs',count:2},{category:'people',count:0}])
  for (const text of ['mugs=1.5','mugs=-1','mugs=65','mugs=true','mugs=','mugs=2\nMUGS=3']) {
    assert.throws(() => parseEndingCounts(text))
  }
})

test('reopened editor shows current or last review stage without starting work', async () => {
  for (const status of ['running', 'needs_attention']) {
    const body = new Element('body')
    const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
    const recipe = {revision:'a'.repeat(64),plan:{project_id:'saved',title:'Film',shots:[],initial_facts:[]},inputs:{library:{references:[]},shots_by_id:{}}}
    const run = {run_id:'b'.repeat(64),revision:recipe.revision,status,phase:'checking_shot',max_attempts:2,shot_id:'one',attempt:1,selected:[{shot_id:'prior',video_sha256:'c'.repeat(64)}],preview:{shot_id:'one',attempt:1,duration_ms:5000}}
    run.reason = 'An object was obscured. Recovery record: /host/production.json'
    run.coverage = {target_duration_ms:60000,planned_shots:7,selected_duration_ms:10000,selected_shots:1}
    const api = {apiURL:path=>path,fetchApi:async(path,options)=> {
      assert.equal(options.method,'GET')
      const data = path==='/comfy/story/films' ? {projects:[]} : path.includes('/runs/') ? run : {recipe,runs:[run],verification_configured:true}
      return {ok:true,json:async()=>structuredClone(data)}
    }}
    await openFilmEditor(api,{properties:{comfy_film_project_id:'saved'}},{},document)
    const prefix = status==='running' ? 'Current' : 'Last'
    assert.ok(body.all().some(x=>x.textContent===`${prefix} stage: Reviewing the current shot`))
    const reason = body.all().find(x=>x.textContent===run.reason)
    assert.equal(reason.parent.tag,'details')
    assert.ok(reason.parent.children.some(x=>x.tag==='summary' && x.textContent==='Run details and recovery information'))
    assert.equal(reason.parent.open,undefined)
    const explanation = body.all().find(x=>x.textContent==='An object was obscured.')
    assert.ok(explanation)
    assert.notEqual(explanation.parent.tag,'details')
    assert.equal(body.all().find(x=>x.tag==='video' && !x.hidden).muted, true)
    const preview = body.all().find(x=>x.tag==='a' && x.textContent==='Preview latest candidate film')
    assert.equal(preview.href, `/comfy/story/films/saved/runs/${run.run_id}/preview`)
    assert.ok(body.all().some(x=>x.textContent?.includes('It may be incomplete or rejected.')))
    assert.ok(body.all().some(x=>x.textContent===filmCoverageText(run)))
    assert.ok(body.all().some(x=>x.textContent==='Candidate preview: 5 seconds. Preview length includes the current candidate, even when it failed review.'))
    await body.all().find(x=>x.tag==='button' && x.textContent==='Close').fire('click')
  }
})


test('new film starts a blank draft without changing the previous project or host run', async () => {
  const body = new Element('body')
  const document = {body, createElement:tag=>new Element(tag), getElementById:id=>body.all().find(x=>x.id===id)}
  const original = {revision:'a'.repeat(64), plan:{project_id:'saved',title:'Original film',shots:[],initial_facts:[]}, inputs:{library:{project_name:'Old cast',references:[]},shots_by_id:{},audio:[{path:'old.wav'}]}}
  const stored = new Map([['saved', structuredClone(original)]])
  const requests = []
  const run = {run_id:'b'.repeat(64),revision:original.revision,status:'running',phase:'rendering_shot',max_attempts:3,selected:[]}
  const api = {apiURL:path=>path, fetchApi:async(path,options)=> {
    requests.push([path,options.method])
    let data
    if(options.method==='POST') {
      assert.equal(path,'/comfy/story/films')
      const payload=JSON.parse(options.body)
      assert.equal(payload.expected_revision,null)
      const recipe={...payload,revision:'c'.repeat(64)}
      stored.set(recipe.plan.project_id,structuredClone(recipe)); data={recipe}
    } else if(path==='/comfy/story/films') data={projects:[{project_id:'saved',title:'Original film'}]}
    else if(path.includes('/runs/')) data=run
    else data={recipe:stored.get(path.split('/').at(-1)),runs:path.endsWith('/saved')?[run]:[],verification_configured:true}
    return {ok:true,json:async()=>structuredClone(data)}
  }}
  const node={properties:{comfy_film_project_id:'saved'}}
  const button=name=>body.all().find(x=>x.tag==='button' && x.textContent===name)
  const control=name=>body.all().find(x=>x.attrs['aria-label']===name)
  await openFilmEditor(api,node,{},document)
  await button('New film').fire('click')
  assert.equal(control('Film title').value,'New film')
  assert.equal(control('Render approach').value,'Continue with references')
  assert.equal(control('Render profile').value,'Reference shot')
  assert.equal(control('Prompt format').value,'H3 automatic v1')
  assert.equal(control('Visible action').value,'')
  assert.equal(control('Open film project').value,'')
  assert.equal(control('Maximum attempts per shot').value,'2')
  assert.equal(node.properties.comfy_film_project_id,'saved')
  assert.equal(body.all().some(x=>x.attrs['aria-label']==='Film run'),false)
  assert.equal(requests.some(([,method])=>method!=='GET'),false)
  control('Film title').value='Fresh story'; await control('Film title').fire('input')
  await button('New film').fire('click')
  assert.equal(control('Film title').value,'Fresh story')
  assert.match(body.all().find(x=>x.attrs.role==='status').textContent,/Save the current draft/)
  control('Visible action').value='A leaf falls.'; await control('Visible action').fire('input')
  control('Starting image (blank uses previous frame)').value='leaf.png'; await control('Starting image (blank uses previous frame)').fire('input')
  await button('Save plan').fire('click')
  const id=node.properties.comfy_film_project_id
  assert.notEqual(id,'saved')
  assert.equal(stored.get(id).plan.title,'Fresh story')
  assert.deepEqual(stored.get(id).inputs.library.references,[])
  assert.equal(stored.get(id).inputs.audio,undefined)
  assert.equal(stored.get(id).inputs.burn_subtitles,false)
  assert.deepEqual(stored.get('saved'),original)
  assert.ok(control('Open film project').children.some(x=>x.value===id && x.textContent==='Fresh story'))
  control('Open film project').value='saved'; await control('Open film project').fire('change')
  assert.equal(control('Film title').value,'Original film')
  assert.equal(control('Maximum attempts per shot').value,'3')
  await button('Close').fire('click')
})


test('new film cannot race a pending save and lose its response', async () => {
  const body=new Element('body')
  const document={body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  const recipe={revision:'a'.repeat(64),plan:{project_id:'saved',title:'Original',shots:[],initial_facts:[]},inputs:{library:{project_name:'Original',references:[]},shots_by_id:{}}}
  let finish, saves=0
  const pending=new Promise(resolve=>{finish=resolve})
  const api={fetchApi:async(path,options)=>{
    if(options.method==='PUT'){saves++;await pending;return {ok:true,json:async()=>({recipe:structuredClone(recipe)})}}
    return {ok:true,json:async()=>path==='/comfy/story/films'?{projects:[]}:{recipe:structuredClone(recipe),runs:[],verification_configured:false}}
  }}
  const button=name=>body.all().find(x=>x.tag==='button' && x.textContent===name)
  await openFilmEditor(api,{properties:{comfy_film_project_id:'saved'}},{},document)
  const saving=button('Save plan').fire('click')
  await Promise.resolve();await Promise.resolve()
  assert.equal(button('New film').disabled,true)
  assert.equal(button('Generate saved plan').disabled,true)
  await button('Generate saved plan').fire('click')
  await button('New film').fire('click');await button('Save plan').fire('click')
  assert.equal(saves,1)
  finish();await saving
  assert.equal(body.all().find(x=>x.attrs['aria-label']==='Film title').value,'Original')
  assert.equal(button('New film').disabled,false)
  await button('Close').fire('click')
})

test('soundtrack controls bind uploaded bytes, persist edits and leave visual recipes unchanged', async () => {
  const body = new Element('body')
  const document = {body, createElement:tag=>new Element(tag), getElementById:id=>body.all().find(x=>x.id===id)}
  let stored, saves=0, inspected=[], audioChanged=false
  const api={apiURL:path=>path, fetchApi:async(path,options)=>{
    let data
    if(path.endsWith('/soundtrack')) {
      inspected.push(JSON.parse(options.body).path)
      data={path:'music/score.wav',sha256:(audioChanged?'c':'b').repeat(64),duration_ms:12000,levels:{status:'measured',peak_dbfs:0,rms_dbfs:-18,silent:false,reaches_full_scale:true}}
    } else if(options.method==='POST' || options.method==='PUT') {
      stored={...JSON.parse(options.body),revision:'a'.repeat(64)};saves++;data={recipe:stored}
    } else if(path==='/comfy/story/films') data={projects:[]}
    else data={recipe:stored,runs:[],verification_configured:false}
    return {ok:true,json:async()=>structuredClone(data)}
  }}
  const node={properties:{}}
  const values={'Story Library':JSON.stringify({project_name:'Sound test',references:[]}), 'What happens next?':'A leaf falls.', 'World / starting frame':'world.png',Variation:42}
  const control=label=>body.all().find(x=>x.attrs['aria-label']===label)
  const button=label=>body.all().find(x=>x.tag==='button' && x.textContent===label)
  await openFilmEditor(api,node,values,document)
  await button('Save plan').fire('click')
  const visuals=structuredClone(stored.inputs.shots_by_id)
  control('Uploaded soundtrack filename').value='music/score.wav';await control('Uploaded soundtrack filename').fire('input')
  await button('Add soundtrack').fire('click')
  assert.deepEqual(inspected,['music/score.wav'])
  assert.ok(body.all().some(x=>x.textContent?.includes('Audio reaches full scale')))
  assert.equal(control('Soundtrack duration (seconds)').value,'10')
  control('Soundtrack volume').value='0.6';await control('Soundtrack volume').fire('input')
  await button('Save plan').fire('click')
  assert.deepEqual(stored.inputs.audio,[{path:'music/score.wav',sha256:'b'.repeat(64),start_ms:0,duration_ms:10000,gain:0.6}])
  assert.deepEqual(stored.inputs.shots_by_id,visuals)
  await button('Close').fire('click');await openFilmEditor(api,node,values,document)
  assert.equal(control('Soundtrack volume').value,'0.6')
  const beforeCheck=structuredClone(stored)
  await button('Check audio levels').fire('click')
  assert.deepEqual(stored,beforeCheck)
  assert.equal(saves,2)
  assert.ok(body.all().some(x=>x.textContent?.includes('Audio reaches full scale')))
  audioChanged=true
  await button('Check audio levels').fire('click')
  assert.match(body.all().find(x=>x.attrs.role==='status').textContent,/soundtrack file has changed/)
  assert.deepEqual(stored,beforeCheck)
  audioChanged=false
  control('Soundtrack start (seconds)').value='1';await control('Soundtrack start (seconds)').fire('input')
  await button('Save plan').fire('click');assert.equal(saves,2)
  control('Soundtrack duration (seconds)').value='9';await control('Soundtrack duration (seconds)').fire('input')
  await button('Save plan').fire('click');assert.equal(saves,3)
  assert.equal(stored.inputs.audio[0].start_ms,1000)
  await button('Remove soundtrack').fire('click');await button('Save plan').fire('click')
  assert.deepEqual(stored.inputs.audio,[])
  assert.deepEqual(stored.inputs.shots_by_id,visuals)
})


test('image previews encode uploaded filenames without accepting external or traversal paths', () => {
  const path = filmImagePreviewPath('cast photos/Mother & Calf #1.png')
  const url = new URL(path, 'https://comfy.example')
  assert.equal(url.pathname, '/view')
  assert.equal(url.searchParams.get('type'), 'input')
  assert.equal(url.searchParams.get('subfolder'), 'cast photos')
  assert.equal(url.searchParams.get('filename'), 'Mother & Calf #1.png')
  for (const value of ['', 'None', null, '../secret.png', '/etc/passwd', 'x/../secret.png', 'x\\secret.png', 'https://external.example/photo.png', '//external.example/photo.png', 'x\n.png']) {
    assert.equal(filmImagePreviewPath(value), null)
  }
})


test('actual editor previews uploaded guides and updates them without saving or generating', async () => {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  let writes = 0
  const api = {apiURL:path=>'/comfy'+path,fetchApi:async(path,options)=>{
    if (options.method !== 'GET') writes++
    return {ok:true,json:async()=>({projects:[]})}
  }}
  await openFilmEditor(api, {properties:{}}, {
    'Story Library':JSON.stringify({project_name:'Preview test',references:[{name:'Boat',role:'Prop',file:'cast/boat.png',note:'A small boat'}]}),
    'World / starting frame':'scenes/dock.png', 'What happens next?':'The boat drifts.', Variation:42,
  }, document)
  const control = label=>body.all().find(x=>x.attrs['aria-label']===label)
  const pictures = ()=>body.all().filter(x=>x.tag==='img')
  assert.equal(pictures().find(x=>x.alt === 'Reference image preview').src, '/comfy'+filmImagePreviewPath('cast/boat.png'))
  const start = pictures().find(x=>x.alt?.startsWith('Starting image'))
  const ending = pictures().find(x=>x.alt?.startsWith('Ending image'))
  assert.equal(start.src, '/comfy'+filmImagePreviewPath('scenes/dock.png'))
  assert.equal(ending.parent.hidden, true)
  control('Ending image (optional)').value='scenes/arrival.png'
  await control('Ending image (optional)').fire('input')
  assert.equal(ending.src, '/comfy'+filmImagePreviewPath('scenes/arrival.png'))
  assert.equal(ending.parent.hidden, false)
  const endingLink=ending.parent.children.find(x=>x.tag==='a')
  assert.equal(endingLink.href, ending.src)
  assert.equal(endingLink.target, '_blank')
  await ending.fire('error')
  assert.equal(ending.hidden, true)
  assert.equal(ending.parent.children.find(x=>x.tag==='p').hidden, false)
  control('Ending image (optional)').value=''
  await control('Ending image (optional)').fire('input')
  assert.equal(ending.parent.hidden, true)
  assert.equal(ending.src, undefined)
  assert.equal(endingLink.href, undefined)
  assert.equal(writes, 0)
  assert.equal(control('Locked seed').value, '42')
})


test('creator state meanings survive save and reopen and can be removed without changing the seed', async () => {
  const body=new Element('body')
  const document={body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  let stored, writes=0
  const api={apiURL:path=>path,fetchApi:async(path,options)=>{
    if (options.method==='POST' || options.method==='PUT') {
      stored={...JSON.parse(options.body),revision:'b'.repeat(64)};writes++
      return {ok:true,json:async()=>({recipe:structuredClone(stored)})}
    }
    return {ok:true,json:async()=>path==='/comfy/story/films'?{projects:[]}:{recipe:structuredClone(stored),runs:[],verification_configured:false}}
  }}
  const node={properties:{}}
  const values={'Story Library':JSON.stringify({project_name:'A box',references:[]}), 'World / starting frame':'box.png','What happens next?':'A closed box is visible.',Variation:123}
  const control=label=>body.all().find(x=>x.attrs['aria-label']===label)
  const button=label=>body.all().find(x=>x.tag==='button' && x.textContent===label)
  await openFilmEditor(api,node,values,document)
  await button('Add state meaning').fire('click')
  for (const [label,value] of [['State key','Box.state'],['State value','closed'],['Observable meaning','The lid covers the entire opening.']]) {
    control(label).value=value;await control(label).fire('input')
  }
  await button('Save plan').fire('click')
  assert.deepEqual(stored.plan.state_definitions,[{key:'Box.state',value:'closed',definition:'The lid covers the entire opening.'}])
  await button('Close').fire('click');await openFilmEditor(api,node,values,document)
  assert.equal(control('Observable meaning').value,'The lid covers the entire opening.')
  assert.equal(control('Locked seed').value,'123')
  await button('Remove state meaning').fire('click');await button('Save plan').fire('click')
  assert.equal(Object.hasOwn(stored.plan,'state_definitions'),false)
  assert.equal(stored.inputs.shots_by_id['shot-1'].variation,123)
  assert.equal(writes,2)
})


test('render approaches preserve legacy settings until an explicit choice', () => {
  const shot={composition:'Continue frame',action:'An unfamiliar object moves.',camera_policy:'Locked frame'}
  const settings={variation:918273,world:'scene.png',sampler:'Native res_multistep'}
  const before=structuredClone({shot,settings})
  assert.equal(filmRenderApproach(shot,settings),'Continue with references')
  setFilmRenderApproach(shot,settings,'Custom settings')
  assert.deepEqual({shot,settings},before)
  setFilmRenderApproach(shot,settings,'Animate starting frame')
  assert.equal(filmRenderApproach(shot,settings),'Animate starting frame')
  assert.equal(settings.render_profile,'Animate frame')
  setFilmRenderApproach(shot,settings,'Compose from references')
  assert.equal(filmRenderApproach(shot,settings),'Compose from references')
  assert.equal(shot.composition,'New composition')
  assert.equal(Object.hasOwn(settings,'render_profile'),false)
  assert.deepEqual(settings,before.settings)
  assert.deepEqual({...shot,composition:'Continue frame'},before.shot)
})

test('an unsupported preset fails atomically without changing the sampler or seed', () => {
  const shot={composition:'New composition'}
  const settings={variation:42,sampler:'SPEED Euler 2-stage'}
  const before=structuredClone({shot,settings})
  assert.throws(()=>setFilmRenderApproach(shot,settings,'Animate starting frame'),/Native or Turbo/)
  assert.deepEqual({shot,settings},before)
  assert.throws(()=>setFilmRenderApproach(shot,settings,'invented'),/supported render approach/)
  assert.deepEqual({shot,settings},before)
})

test('switching Turbo 8 animation to reference composition preserves the requested sampler and seed', () => {
  const shot={composition:'Continue frame'}
  const settings={variation:42,world:'scene.png',sampler:'Turbo 8-step',render_profile:'Animate frame'}
  setFilmRenderApproach(shot,settings,'Compose from references')
  assert.equal(filmRenderApproach(shot,settings),'Compose from references')
  assert.equal(settings.sampler,'Turbo 8-step')
  assert.equal(settings.variation,42)
  assert.equal(settings.world,'scene.png')
  for (const sampler of ['Native res_multistep','Turbo 4-step','SPEED Euler 2-stage']) {
    const other={sampler,variation:42}
    setFilmRenderApproach({},other,'Compose from references')
    assert.equal(other.sampler,sampler)
  }
})

test('editor approach selection persists a coherent pair and added shots inherit it', async () => {
  const body=new Element('body')
  const document={body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  let stored,mutations=0
  const api={apiURL:path=>path,fetchApi:async(path,options)=>{
    if(options.method==='POST'||options.method==='PUT') {mutations++;stored=JSON.parse(options.body);return {ok:true,json:async()=>({recipe:{...stored,revision:'a'.repeat(64)}})}}
    return {ok:true,json:async()=>path==='/comfy/story/films'?{projects:[]}:{verification_configured:true}}
  }}
  const control=name=>body.all().find(x=>x.attrs['aria-label']===name)
  const button=name=>body.all().find(x=>x.tag==='button'&&x.textContent===name)
  await openFilmEditor(api,{properties:{}},{'Story Library':JSON.stringify({project_name:'Existing draft',references:[]}),'What happens next?':'A shape moves.','World / starting frame':'scene.png',Variation:42},document)
  assert.equal(control('Render approach').value,'Continue with references')
  assert.equal(mutations,0)
  control('Render approach').value='Compose from references';await control('Render approach').fire('change')
  assert.equal(control('Composition').value,'New composition')
  assert.equal(control('Render profile').value,'Reference shot')
  await button('Add shot').fire('click')
  assert.deepEqual(body.all().filter(x=>x.attrs['aria-label']==='Render approach').map(x=>x.value),['Compose from references','Compose from references'])
  await button('Save plan').fire('click')
  assert.equal(mutations,1)
  assert.equal(stored.inputs.shots_by_id['shot-1'].variation,42)
  assert.equal(stored.inputs.shots_by_id['shot-1'].world,'scene.png')
  assert.ok(stored.plan.shots.every(x=>x.composition==='New composition'))
  assert.ok(Object.values(stored.inputs.shots_by_id).every(x=>!Object.hasOwn(x,'render_profile')))
})

test('customer can add named cast in the editor and duplicate or unsafe names do not mutate it', async () => {
  const body = new Element('body')
  const document = {body, createElement:tag=>new Element(tag), getElementById:id=>body.all().find(x=>x.id===id)}
  let stored
  const api = {apiURL:path=>path, fetchApi:async(path,options)=>{
    let data
    if (options.method==='POST' || options.method==='PUT') {
      stored = {...JSON.parse(options.body),revision:'a'.repeat(64)}; data={recipe:stored}
    } else if (path==='/comfy/story/films') data={projects:[]}
    else data={recipe:stored,runs:[],verification_configured:false}
    return {ok:true,json:async()=>structuredClone(data)}
  }}
  const control = label=>body.all().find(x=>x.attrs['aria-label']===label)
  const button = label=>body.all().find(x=>x.tag==='button' && x.textContent===label)
  const node={properties:{}}
  await openFilmEditor(api,node,{'Story Library':JSON.stringify({project_name:'Forest',references:[]}),Variation:42},document)
  control('New reference name').value='  Acorn  '; await control('New reference name').fire('input')
  await button('Add reference').fire('click')
  assert.equal(control('New reference name').value,'')
  assert.ok(control('Upload Acorn'))
  control('Reference image').value='acorn.png'; await control('Reference image').fire('input')
  control('Reference role').value='Prop'; await control('Reference role').fire('change')
  control('Present reference names').value='Acorn'; await control('Present reference names').fire('input')
  control('Visible action').value='Acorn rolls down a slope.'; await control('Visible action').fire('input')
  control('Starting image (blank uses previous frame)').value='forest.png'; await control('Starting image (blank uses previous frame)').fire('input')
  for (const name of ['acorn','../escape','2Acorn','two names','x'.repeat(65)]) {
    control('New reference name').value=name; await control('New reference name').fire('input')
    await button('Add reference').fire('click')
    assert.match(body.all().find(x=>x.attrs.role==='status').textContent,/Reference name/)
    assert.equal(body.all().filter(x=>x.attrs['aria-label']==='Reference image').length,1)
  }
  control('New reference name').value=''; await control('New reference name').fire('input')
  await button('Save plan').fire('click')
  assert.equal(stored.inputs.library.references[0].name,'Acorn')
  assert.equal(stored.inputs.library.references[0].file,'acorn.png')
  assert.deepEqual(stored.plan.shots[0].present,['Acorn'])
  await button('Close').fire('click')
  await openFilmEditor(api,node,{},document)
  assert.ok(control('Upload Acorn'))
  assert.equal(control('Present reference names').value,'Acorn')
  await button('Close').fire('click')
})

test('staging requires a separate opening and bounded image budget without copying future action', () => {
  const shot = {action:'Open the parcel', composition:'New composition'}
  const settings = {sampler:'Turbo 8-step', variation:42}
  setFilmRenderApproach(shot, settings, 'Stage then animate')
  assert.equal(settings.opening_prompt, '')
  assert.equal(settings.opening_attempts, 1)
  assert.equal(settings.intent, 'New Scene')
  assert.equal(settings.render_profile, 'Animate frame')
  assert.equal(shot.composition, 'Continue frame')
  assert.equal(filmRenderApproach(shot, settings), 'Stage then animate')
  settings.opening_prompt = 'The unopened parcel on a table'
  setFilmRenderApproach(shot, settings, 'Animate starting frame')
  assert.equal(Object.hasOwn(settings, 'opening_prompt'), false)
  assert.equal(Object.hasOwn(settings, 'opening_attempts'), false)
  assert.equal(filmRunStage({phase:'checking_opening'}), 'Checking the opening image')
})

test('switching away from staging clears the preceding-scene option', () => {
  const shot={composition:'Continue frame'}
  const settings={render_profile:'Animate frame',opening_prompt:'A wider view',opening_use_previous_scene:true}
  setFilmRenderApproach(shot,settings,'Animate starting frame')
  assert.equal(Object.hasOwn(settings,'opening_use_previous_scene'),false)
})

test('optional story intent survives save and reopen, rejects partial data and clears without seed changes', async () => {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  let stored = {revision:'a'.repeat(64),plan:{project_id:'intent',title:'A short action',target_duration_ms:2500,initial_facts:[],shots:[{shot_id:'one',duration_ms:2500,purpose:'Act',action:'A leaf turns.',present:[]}]},inputs:{library:{references:[]},shots_by_id:{one:{variation:42,world:'leaf.png'}}}}
  let saves = 0
  const api = {apiURL:path=>path,fetchApi:async(path,options)=>{
    if (options.method === 'PUT') { stored={...JSON.parse(options.body),revision:'b'.repeat(64)}; saves++ }
    return {ok:true,json:async()=>path==='/comfy/story/films' ? {projects:[]} : {recipe:structuredClone(stored),runs:[],verification_configured:false,impact:{reusable_prefix:['one'],requires_generation_review:[],narrative_review_changed:true}}}
  }}
  const node={properties:{comfy_film_project_id:'intent'}}
  const control=label=>body.all().find(x=>x.attrs['aria-label']===label)
  const button=label=>body.all().find(x=>x.tag==='button' && x.textContent===label)
  await openFilmEditor(api,node,{},document)
  control('Intended goal').value='Reach the light.'; await control('Intended goal').fire('input')
  await button('Save plan').fire('click')
  assert.equal(saves,0)
  for (const field of ['obstacle','decision','outcome']) {
    control('Intended '+field).value='Description of '+field
    await control('Intended '+field).fire('input')
  }
  await button('Save plan').fire('click')
  assert.equal(saves,1)
  assert.match(body.all().find(x=>x.attrs.role==='status').textContent,/Whole-film review is required/)
  assert.equal(stored.plan.narrative.goal,'Reach the light.')
  assert.equal(stored.inputs.shots_by_id.one.variation,42)
  await button('Close').fire('click')
  await openFilmEditor(api,node,{},document)
  assert.equal(control('Intended goal').value,'Reach the light.')
  for (const field of ['goal','obstacle','decision','outcome']) {
    control('Intended '+field).value=''; await control('Intended '+field).fire('input')
  }
  await button('Save plan').fire('click')
  assert.equal(saves,2)
  assert.match(body.all().find(x=>x.attrs.role==='status').textContent,/Whole-film review is required/)
  assert.equal('narrative' in stored.plan,false)
  assert.equal(stored.inputs.shots_by_id.one.variation,42)
})

for (const fails of [false, true]) test(`project loading cannot launch or overwrite the previous film (${fails ? 'failed load' : 'successful load'})`, async () => {
  const body = new Element('body')
  const document = {body, createElement: tag => new Element(tag), getElementById: id => body.all().find(x => x.id === id)}
  const recipe = id => ({revision: id === 'first' ? 'a'.repeat(64) : 'b'.repeat(64), plan: {project_id:id, title:id, shots:[], initial_facts:[]}, inputs:{library:{references:[]}, shots_by_id:{}}})
  let finish
  const pending = new Promise(resolve => { finish = resolve })
  const mutations = [], reads = []
  const api = {fetchApi: async (path, options) => {
    if (options.method !== 'GET') { mutations.push({path,body:JSON.parse(options.body)}); return {ok:false,json:async()=>({error:'Recorded submission'})} }
    reads.push(path)
    if (path.endsWith('/second')) { await pending; if (fails) return {ok:false,json:async()=>({error:'Project unavailable'})} }
    return {ok:true,json:async()=>path === '/comfy/story/films' ? {projects:[{project_id:'first',title:'first'},{project_id:'second',title:'second'}]} : {recipe:recipe(path.endsWith('/second') ? 'second' : 'first'), runs:[], verification_configured:true}}
  }}
  const node = {properties:{comfy_film_project_id:'first'}}
  const control = label => body.all().find(x => x.attrs['aria-label'] === label)
  const button = label => body.all().find(x => x.tag === 'button' && x.textContent === label)
  await openFilmEditor(api,node,{},document)
  const chooser = control('Open film project')
  chooser.value = 'second'
  const loading = chooser.fire('change')
  await Promise.resolve(); await Promise.resolve()
  for (const label of ['Save plan','Generate saved plan','Resume selected run','New film']) {
    assert.equal(button(label).disabled,true)
    await button(label).fire('click')
  }
  assert.equal(chooser.disabled,true)
  assert.equal(control('Film title').parent.parent.inert,true)
  assert.equal(button('Close').disabled,undefined)
  assert.equal(mutations.length,0)
  finish(); await loading
  const expected = fails ? 'first' : 'second'
  assert.equal(node.properties.comfy_film_project_id,expected)
  assert.equal(chooser.value,expected)
  assert.equal(control('Film title').value,expected)
  assert.equal(control('Film title').parent.parent.inert,false)
  assert.equal(button('Generate saved plan').disabled,false)
  const status = body.all().find(x => x.attrs.role === 'status').textContent
  assert.match(status, fails ? /Project unavailable/ : /Opened saved film project/)
  await button('Generate saved plan').fire('click')
  assert.equal(mutations.length,1)
  assert.equal(mutations[0].path,`/comfy/story/films/${expected}/runs`)
  assert.equal(mutations[0].body.revision,recipe(expected).revision)
  await button('Close').fire('click')
})

for (const malformed of [false,true]) test(`recipe import is isolated from project saves and launches (${malformed ? 'invalid' : 'valid'})`, async () => {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  const stored = {revision:'a'.repeat(64),plan:{project_id:'first',title:'Original',initial_facts:[],shots:[]},inputs:{library:{references:[]},shots_by_id:{}}}
  let finish
  const reading = new Promise(resolve=>{finish=resolve}), requests=[]
  const api = {fetchApi:async(path,options)=>{
    requests.push({path,method:options.method})
    assert.equal(options.method,'GET')
    return {ok:true,json:async()=>path==='/comfy/story/films'?{projects:[]}:{recipe:structuredClone(stored),runs:[],verification_configured:false}}
  }}
  const button = label=>body.all().find(x=>x.tag==='button' && x.textContent===label)
  const control = label=>body.all().find(x=>x.attrs['aria-label']===label)
  await openFilmEditor(api,{properties:{comfy_film_project_id:'first'}},{},document)
  const input = control('Import film recipe')
  input.files=[{size:100,text:async()=>{await reading;return malformed?'not JSON':JSON.stringify({...stored,plan:{...stored.plan,title:'Imported'}})}}]
  const importing=input.fire('change')
  await Promise.resolve();await Promise.resolve()
  const before=requests.length
  for(const label of ['Save plan','Generate saved plan','New film']) {assert.equal(button(label).disabled,true);await button(label).fire('click')}
  control('Open film project').value='second';await control('Open film project').fire('change')
  assert.equal(requests.length,before)
  finish();await importing
  assert.equal(control('Film title').value,malformed?'Original':'Imported')
  assert.equal(button('Save plan').disabled,false)
  assert.equal(input.disabled,false)
  assert.equal(input.value,'')
  await button('Close').fire('click')
})

test('editor distinguishes prose and state meanings from declared state checks as the draft changes', async () => {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  const recipe = {revision:'a'.repeat(64),plan:{project_id:'coverage',title:'A repair',target_duration_ms:5000,initial_facts:[],state_definitions:[{key:'lamp.light',value:'off',definition:'The bulb is visibly dark.'}],narrative:{goal:'Restore light.',obstacle:'The light is off.',decision:'Repair it.',outcome:'The light shines.'},shots:[{shot_id:'one',duration_ms:5000,purpose:'Repair',action:'The lamp turns on.',present:[],requires:[],effects:[]}]},inputs:{library:{references:[]},shots_by_id:{one:{variation:42,world:'lamp.png',opening_prompt:'A dark lamp.'}}}}
  const original = structuredClone(recipe)
  let writes = 0
  const api = {apiURL:path=>path,fetchApi:async(path,options)=>{
    if (options.method !== 'GET') writes++
    return {ok:true,json:async()=>path==='/comfy/story/films' ? {projects:[]} : {recipe:structuredClone(recipe),runs:[],verification_configured:true}}
  }}
  const node = {properties:{comfy_film_project_id:'coverage'}}
  const control = label=>body.all().find(x=>x.attrs['aria-label']===label)
  const message = ()=>control('Declared state checks').textContent
  await openFilmEditor(api,node,{},document)
  assert.match(message(),/no explicit starting or ending facts/)
  assert.match(message(),/General action and visibility review can still run/)
  for (const label of ['Initial facts (key=value per line)','Required starting facts','Visible ending facts']) {
    control(label).value = 'lamp.light=off'; await control(label).fire('input')
    assert.match(message(),/State checks use your declared facts/)
    assert.match(message(),/do not verify every detail/)
    control(label).value = ''; await control(label).fire('input')
    assert.match(message(),/no explicit starting or ending facts/)
  }
  assert.equal(writes,0)
  assert.deepEqual(recipe,original)
  await body.all().find(x=>x.tag==='button' && x.textContent==='Close').fire('click')
})

test('stopped opening review shows observed facts and images without changing the recipe or approving', async () => {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  const recipe = {revision:'a'.repeat(64),plan:{project_id:'saved',title:'Film',initial_facts:[],shots:[{shot_id:'one',duration_ms:5000,purpose:'Act',action:'A leaf turns.',present:[]}]},inputs:{library:{references:[]},shots_by_id:{one:{variation:42}}}}
  const run = {run_id:'b'.repeat(64),revision:recipe.revision,status:'needs_attention',phase:'checking_opening',max_attempts:2,shot_id:'one',opening_attempt:2,selected:[],opening_candidates:[{attempt:1,image:'staged/fitted.png',raw_image:'staged/raw.png',fitted:true,state:{conditions:{'Leaf.color':{expected:'green',observed:'brown',evidence:'Brown surface.'}}},visibility:{observations:{Leaf:{extent:'entire',evidence:'Whole leaf visible.'}}}},{attempt:2,unavailable:'Opening evidence is incomplete, missing or changed.'}]}
  const calls = []
  const api = {apiURL:path=>path,fetchApi:async(path,options)=>{calls.push([path,options.method]);return {ok:true,json:async()=>path==='/comfy/story/films'?{projects:[]}:path.includes('/runs/')?run:{recipe:structuredClone(recipe),runs:[run],verification_configured:true}}}}
  run.opening_candidates[0].visibility.requirements = {Crate:'absent'}
  run.opening_candidates[0].visibility.observations.Crate = {extent:'partial',evidence:'Corner at the picture edge.'}
  await openFilmEditor(api,{properties:{comfy_film_project_id:'saved'}},{},document)
  assert.ok(body.all().some(x=>x.textContent==='Leaf.color: expected green; observed brown. Brown surface.'))
  assert.ok(body.all().some(x=>x.textContent==='Leaf: entire. Whole leaf visible.'))
  assert.ok(body.all().some(x=>x.textContent==='Crate: expected absent; observed partial. Corner at the picture edge.'))
  assert.equal(body.all().find(x=>x.tag==='img' && x.alt==='Assessed H3 opening crop').src,'/view?type=input&subfolder=staged&filename=fitted.png')
  assert.equal(body.all().find(x=>x.tag==='a' && x.textContent==='Open uncropped source').href,'/view?type=input&subfolder=staged&filename=raw.png')
  assert.ok(body.all().some(x=>x.textContent?.includes('this image is not approved')))
  assert.ok(body.all().some(x=>x.textContent==='Opening evidence is incomplete, missing or changed.'))
  assert.ok(calls.every(([,method])=>!method || method==='GET'))
  await body.all().find(x=>x.tag==='button' && x.textContent==='Close').fire('click')
})

test('refinement mode survives save and reopen without changing the locked seed', async () => {
  const body=new Element('body')
  const document={body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  let stored={revision:'a'.repeat(64),plan:{project_id:'refine',title:'A prepared scene',target_duration_ms:5000,initial_facts:[],shots:[{shot_id:'one',duration_ms:5000,purpose:'Observe',action:'A leaf turns.',present:[],composition:'Continue frame'}]},inputs:{library:{references:[]},shots_by_id:{one:{variation:72,world:'leaf.png',opening_prompt:'Blend the prepared leaf with its background',opening_mode:'Compose',opening_attempts:1,render_profile:'Animate frame',intent:'New Scene'}}}}
  const api={apiURL:path=>path,fetchApi:async(path,options)=>{
    if(options.method==='PUT')stored={...JSON.parse(options.body),revision:'b'.repeat(64)}
    return {ok:true,json:async()=>path==='/comfy/story/films'?{projects:[]}:{recipe:structuredClone(stored),runs:[],verification_configured:false}}
  }}
  const control=label=>body.all().find(x=>x.attrs['aria-label']===label)
  const button=label=>body.all().find(x=>x.tag==='button'&&x.textContent===label)
  const node={properties:{comfy_film_project_id:'refine'}}
  await openFilmEditor(api,node,{},document)
  assert.equal(control('Opening image mode').value,'Compose')
  control('Opening image mode').value='Refine';await control('Opening image mode').fire('change')
  await button('Save plan').fire('click')
  assert.equal(stored.inputs.shots_by_id.one.opening_mode,'Refine')
  assert.equal(stored.inputs.shots_by_id.one.variation,72)
  await button('Close').fire('click');await openFilmEditor(api,node,{},document)
  assert.equal(control('Opening image mode').value,'Refine')
  control('Render approach').value='Animate starting frame';await control('Render approach').fire('change')
  await button('Save plan').fire('click')
  assert.equal(Object.hasOwn(stored.inputs.shots_by_id.one,'opening_mode'),false)
  assert.equal(stored.inputs.shots_by_id.one.variation,72)
  await button('Close').fire('click')
})


test('uploaded starting-state failure shows evidence without edits or generation', async () => {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  const recipe = {revision:'a'.repeat(64),plan:{project_id:'parcel',title:'A delivery',shots:[],initial_facts:[]},inputs:{library:{references:[]},shots_by_id:{}}}
  const run = {run_id:'b'.repeat(64),revision:recipe.revision,status:'needs_attention',phase:'checking_inputs',max_attempts:2,selected:[],starting_state_review:{kind:'uploaded',image:'checks/parcel.png',state:{conditions:{'Parcel.state':{expected:'sealed',observed:null,evidence:'Seal is obscured.'}}}}}
  const api = {apiURL:path=>path,fetchApi:async(path,options)=> {
    assert.equal(options.method,'GET')
    return {ok:true,json:async()=>structuredClone(path==='/comfy/story/films' ? {projects:[]} : path.includes('/runs/') ? run : {recipe,runs:[run],verification_configured:true})}
  }}
  await openFilmEditor(api,{properties:{comfy_film_project_id:'parcel'}},{},document)
  assert.ok(body.all().some(x=>x.textContent==='Starting image check'))
  assert.equal(body.all().find(x=>x.alt==='Assessed starting image').src,'/view?type=input&subfolder=checks&filename=parcel.png')
  assert.ok(body.all().some(x=>x.textContent==='Parcel.state: expected sealed; observed uncertain. Seal is obscured.'))
  assert.ok(!body.all().some(x=>x.textContent==='Opening image attempt undefined'))
  await body.all().find(x=>x.tag==='button' && x.textContent==='Close').fire('click')
})

test('first cut is the editor default without a reviewer; legacy resume retains checks', async () => {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  const recipe = {revision:'a'.repeat(64),plan:{project_id:'delivery',title:'Leaf',target_duration_ms:5000,initial_facts:[],shots:[{shot_id:'one',duration_ms:5000,purpose:'Turn',action:'A leaf turns.',present:[]}]},inputs:{library:{references:[]},shots_by_id:{one:{variation:42,world:'leaf.png'}}}}
  let run = {run_id:'b'.repeat(64),revision:recipe.revision,status:'needs_attention',max_attempts:2}
  const posts = []
  const api = {apiURL:path=>path,fetchApi:async(path,options)=>{
    if (options.method === 'POST') { posts.push(JSON.parse(options.body)); run = {...run,mode:posts.at(-1).mode,status:'draft_ready',phase:'complete',rendered:[{shot_id:'one',video_sha256:'c'.repeat(64)}]} }
    return {ok:true,json:async()=>path==='/comfy/story/films'?{projects:[]}:path.includes('/runs')?run:{recipe:structuredClone(recipe),runs:[run],verification_configured:false}}
  }}
  const button = label=>body.all().find(x=>x.tag==='button' && x.textContent===label)
  await openFilmEditor(api,{properties:{comfy_film_project_id:'delivery'}},{},document)
  assert.equal(body.all().find(x=>x.attrs['aria-label']==='Generation mode').value,'First cut')
  assert.ok(!body.all().some(x=>x.textContent?.includes('host must configure')))
  await button('Resume selected run').fire('click')
  assert.equal(posts[0].mode,'checked')
  assert.equal(posts[0].run_id,'b'.repeat(64))
  await button('Generate saved plan').fire('click')
  assert.equal(posts[1].mode,'first_cut')
  assert.equal(posts[1].run_id,undefined)
  assert.equal(button('Pause after current shot').hidden,true)
  assert.equal(button('Resume selected run').hidden,true)
  assert.ok(body.all().some(x=>x.tag==='a' && x.textContent==='Open assembled film'))
  assert.ok(body.all().some(x=>x.tag==='video' && x.src.endsWith('c'.repeat(64))))
  assert.equal(filmRunStage(run),'First cut ready to watch and edit')
  assert.equal(filmCoverageText({...run,coverage:{planned_duration_ms:5000,planned_shots:1}}),'5 seconds rendered (1 shots).')
  await button('Close').fire('click')
})

for (const fails of [false,true]) test(`slow soundtrack attachment prevents an incomplete save or launch (${fails ? 'failure' : 'success'})`, async () => {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  let stored = {revision:'a'.repeat(64),plan:{project_id:'long-film',title:'Long film',target_duration_ms:180000,initial_facts:[],shots:Array.from({length:18},(_,i)=>({shot_id:`shot-${i}`,duration_ms:10000,purpose:'Journey',action:'A rover moves.',present:[]}))},inputs:{library:{references:[]},shots_by_id:Object.fromEntries(Array.from({length:18},(_,i)=>[`shot-${i}`,{variation:i+1,world:'rover.png'}]))}}
  let finish, inspections=0
  const pending = new Promise(resolve=>{finish=resolve})
  const writes=[]
  const api={apiURL:path=>path,fetchApi:async(path,options)=>{
    if(path.endsWith('/soundtrack')) {
      inspections++;await pending
      return {ok:!fails,json:async()=>fails?{error:'Audio unavailable'}:{path:'score.flac',sha256:'c'.repeat(64),duration_ms:180000}}
    }
    if(options.method!=='GET') {writes.push(path);stored={...JSON.parse(options.body),revision:'b'.repeat(64)}}
    return {ok:true,json:async()=>path==='/comfy/story/films'?{projects:[]}:{recipe:structuredClone(stored),runs:[],verification_configured:false}}
  }}
  const button=label=>body.all().find(x=>x.tag==='button' && x.textContent===label)
  const control=label=>body.all().find(x=>x.attrs['aria-label']===label)
  await openFilmEditor(api,{properties:{comfy_film_project_id:'long-film'}},{},document)
  control('Uploaded soundtrack filename').value='score.flac';await control('Uploaded soundtrack filename').fire('input')
  const attaching=button('Add soundtrack').fire('click')
  await Promise.resolve()
  for(const label of ['Save plan','Generate saved plan','New film']) {assert.equal(button(label).disabled,true);await button(label).fire('click')}
  await button('Add soundtrack').fire('click')
  assert.equal(inspections,1)
  assert.deepEqual(writes,[])
  assert.equal(control('Film title').parent.parent.inert,true)
  finish();await attaching
  assert.equal(button('Save plan').disabled,false)
  assert.equal(control('Film title').parent.parent.inert,false)
  if(fails) assert.match(body.all().find(x=>x.attrs.role==='status').textContent,/Audio unavailable/)
  else {
    assert.equal(control('Soundtrack duration (seconds)').value,'180')
    await button('Save plan').fire('click')
    assert.equal(stored.inputs.audio[0].duration_ms,180000)
    assert.equal(writes.length,1)
    assert.equal(stored.inputs.shots_by_id['shot-17'].variation,18)
  }
  await button('Close').fire('click')
})

function workspaceFixture() {
  const body = new Element('body')
  const document = {body,createElement:tag=>new Element(tag),getElementById:id=>body.all().find(x=>x.id===id)}
  const recipe = {revision:'a'.repeat(64),plan:{project_id:'workspace',title:'A journey',initial_facts:[],shots:[
    {shot_id:'one',duration_ms:5000,purpose:'Arrival',action:'@Aiko arrives.',present:['Aiko']},
    {shot_id:'two',duration_ms:5000,purpose:'Departure',action:'@Aiko leaves.',present:[]},
  ]},inputs:{library:{project_name:'A journey',references:[{name:'Aiko',role:'Character',file:'aiko.png',note:'Blue coat.'}]},shots_by_id:{one:{variation:11},two:{variation:22}}}}
  const run = {run_id:'b'.repeat(64),revision:recipe.revision,status:'draft_ready',mode:'first_cut',phase:'complete',rendered:[{shot_id:'one',video_sha256:'c'.repeat(64)},{shot_id:'two',video_sha256:'d'.repeat(64)}]}
  const writes = []
  const api = {apiURL:path=>path,fetchApi:async(path,options)=>{
    if (options.method !== 'GET') { writes.push(JSON.parse(options.body)); return {ok:true,json:async()=>({recipe:{...writes.at(-1),revision:'e'.repeat(64)},impact:{reusable_prefix:['one'],requires_generation_review:['two']}})} }
    return {ok:true,json:async()=>structuredClone(path==='/comfy/story/films'?{projects:[]}:path.includes('/runs/')?run:{recipe,runs:[run],verification_configured:false})}
  }}
  const control = label=>body.all().find(x=>x.attrs['aria-label']===label)
  const button = label=>body.all().find(x=>x.tag==='button' && x.textContent===label)
  return {body,document,recipe,run,writes,api,control,button,node:{properties:{comfy_film_project_id:'workspace'}}}
}

test('shot navigation and workspace sections preserve draft inputs without saves or new seeds', async () => {
  const f = workspaceFixture(); await openFilmEditor(f.api,f.node,{},f.document)
  const before = structuredClone(f.recipe)
  f.control('Visible action').value='@Aiko pauses.'; await f.control('Visible action').fire('input')
  await f.control('Edit shot 2: Departure').fire('click')
  const cards=()=>f.body.all().filter(x=>x.className==='comfy-film-shot')
  assert.deepEqual(cards().map(x=>x.hidden),[true,false])
  await f.button('Cast & world').fire('click')
  assert.equal(f.button('Cast & world').attrs['aria-pressed'],'true')
  await f.button('Shots').fire('click')
  await f.control('Edit shot 1: Arrival').fire('click')
  assert.equal(f.control('Visible action').value,'@Aiko pauses.')
  assert.deepEqual(cards().map(x=>x.hidden),[false,true])
  assert.deepEqual(f.recipe,before)
  assert.equal(f.writes.length,0)
  await f.button('Save plan').fire('click')
  assert.equal(f.writes[0].plan.shots[0].action,'@Aiko pauses.')
  assert.deepEqual(f.writes[0].inputs,before.inputs)
  assert.ok(f.body.all().some(x=>x.textContent==='Needs generation review'))
  await f.button('Close').fire('click')
})

test('playback survives status refreshes and switches explicitly between a take and the film', async () => {
  const f=workspaceFixture(); await openFilmEditor(f.api,f.node,{},f.document)
  const player=f.body.all().find(x=>x.tag==='video')
  let source=player.src, replacements=0
  Object.defineProperty(player,'src',{get:()=>source,set:x=>{source=x;replacements++}})
  player.muted=false
  f.control('Generation mode').value='First cut';await f.control('Generation mode').fire('change')
  assert.equal(replacements,0)
  assert.equal(player.muted,false)
  await f.button('Watch assembled film').fire('click')
  assert.match(player.src,/\/runs\/[b]+\/video$/)
  assert.equal(player.muted,false)
  await f.control('Generation mode').fire('change')
  assert.equal(replacements,1)
  await f.control('Edit shot 2: Departure').fire('click')
  assert.ok(player.src.endsWith('d'.repeat(64)))
  assert.equal(player.muted,true)
  assert.equal(replacements,2)
  await f.button('Close').fire('click')
})

test('reference chips update the saved roster and exact-name edits update chips', async () => {
  const f=workspaceFixture(); await openFilmEditor(f.api,f.node,{},f.document)
  await f.control('Edit shot 2: Departure').fire('click')
  const chip=f.control('Include @Aiko in shot 2'); chip.checked=true;await chip.fire('change')
  const second=f.body.all().filter(x=>x.className==='comfy-film-shot')[1]
  const names=second.all().find(x=>x.attrs['aria-label']==='Present reference names')
  assert.equal(names.value,'Aiko')
  names.value='';await names.fire('input');assert.equal(chip.checked,false)
  chip.checked=true;await chip.fire('change')
  await f.button('Save plan').fire('click')
  assert.deepEqual(f.writes[0].plan.shots[1].present,['Aiko'])
  assert.equal(f.writes[0].inputs.shots_by_id.two.variation,22)
  await f.button('Close').fire('click')
})

test('saving reveals an invalid field even after switching sections and shots', async () => {
  const f=workspaceFixture(); await openFilmEditor(f.api,f.node,{},f.document)
  f.control('Locked seed').value='bad';await f.control('Locked seed').fire('input')
  await f.control('Edit shot 2: Departure').fire('click')
  await f.button('Soundtrack').fire('click')
  await f.button('Save plan').fire('click')
  assert.equal(f.writes.length,0)
  assert.equal(f.button('Shots').attrs['aria-pressed'],'true')
  assert.equal(f.body.all().find(x=>x.className==='comfy-film-shot').hidden,false)
  assert.ok(f.control('Locked seed').validationMessage)
  await f.button('Close').fire('click')
})

for (const fails of [false,true]) test(`audio file upload preserves the film while pending (${fails?'failure':'success'})`, async () => {
  const f=workspaceFixture(), fallback=f.api.fetchApi
  let finish, uploads=0
  const pending=new Promise(resolve=>{finish=resolve})
  f.api.fetchApi=async(path,options)=>{
    if (!path.endsWith('/upload-soundtrack')) return fallback(path,options)
    uploads++;assert.ok(options.body instanceof FormData);assert.equal(options.body.get('audio').name,'music.wav')
    await pending
    return {ok:!fails,json:async()=>({path:'story-audio-new.wav',sha256:'f'.repeat(64),duration_ms:7000})}
  }
  await openFilmEditor(f.api,f.node,{},f.document)
  await f.button('Soundtrack').fire('click')
  const input=f.control('Upload soundtrack');input.files=[new File(['audio fixture'],'music.wav')]
  const upload=input.fire('change');await Promise.resolve()
  assert.equal(f.button('Save plan').disabled,true)
  await f.button('Save plan').fire('click');await input.fire('change')
  assert.equal(uploads,1);assert.equal(f.writes.length,0)
  finish();await upload
  assert.equal(f.button('Save plan').disabled,false)
  if (!fails) {
    assert.equal(f.control('Soundtrack duration (seconds)').value,'7')
    await f.button('Save plan').fire('click')
    assert.equal(f.writes[0].inputs.audio[0].path,'story-audio-new.wav')
    assert.deepEqual(f.writes[0].inputs.shots_by_id,f.recipe.inputs.shots_by_id)
  } else {
    assert.match(f.body.all().find(x=>x.attrs.role==='status').textContent,/upload failed/)
    assert.equal(f.control('Soundtrack duration (seconds)'),undefined)
  }
  await f.button('Close').fire('click')
})


test('closing a dirty draft requires an explicit discard and never saves it silently', async () => {
  const f=workspaceFixture();await openFilmEditor(f.api,f.node,{},f.document)
  f.control('Visible action').value='A different action.';await f.control('Visible action').fire('input')
  await f.button('Close').fire('click')
  assert.ok(f.document.getElementById('comfy-film-editor'))
  assert.equal(f.button('Discard draft and close').hidden,false)
  assert.equal(f.writes.length,0)
  await f.button('Discard draft and close').fire('click')
  assert.equal(f.document.getElementById('comfy-film-editor'),undefined)
  await openFilmEditor(f.api,f.node,{},f.document)
  assert.equal(f.control('Visible action').value,'@Aiko arrives.')
  assert.equal(f.writes.length,0)
  await f.button('Close').fire('click')
})

import { filmReferenceReviewInputs } from '../../integrations/comfy_story/web/film_editor.mjs'

test('reference review uses declared cast, preserves library order and leaves recipes intact', () => {
  const shot = {present:['moonblade','AIKO']}
  const settings = {sampler:'Turbo 8-step'}
  const library = {references:[{name:'Aiko',file:'a.png',note:'keep face'}, {name:'Forest',file:'f.png'}, {name:'Moonblade',file:'m.png'}]}
  const before = JSON.stringify({shot,settings,library})
  assert.deepEqual(filmReferenceReviewInputs(shot,settings,library), {sampler:'Turbo 8-step',references:[{name:'Aiko',file:'a.png'},{name:'Moonblade',file:'m.png'}]})
  assert.equal(JSON.stringify({shot,settings,library}),before)
  assert.deepEqual(filmReferenceReviewInputs({}, {}, library), {sampler:'Native res_multistep', references:[]})
})
