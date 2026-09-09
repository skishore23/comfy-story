export const filmEditorStyles = `
#comfy-film-editor{--film-bg:#14171d;--film-panel:#1d222b;--film-line:#36404d;--film-muted:#b0bbca;--film-accent:#a6e4cb;box-sizing:border-box;width:min(1400px,96vw);height:94vh;max-height:94vh;padding:0;overflow:auto;background:var(--film-bg);color:#eef2f7;border:1px solid var(--film-line);border-radius:18px;font:14px/1.5 system-ui,sans-serif;color-scheme:dark}
#comfy-film-editor::backdrop{background:#080c12c9}
#comfy-film-editor *{box-sizing:border-box}
#comfy-film-editor [hidden]{display:none!important}
#comfy-film-editor h2,#comfy-film-editor h3,#comfy-film-editor h4,#comfy-film-editor p{margin:0 0 12px}
#comfy-film-editor h2{font-size:23px;letter-spacing:-.5px}
#comfy-film-editor h3{font-size:17px}
#comfy-film-editor p,#comfy-film-editor small{color:var(--film-muted)}
#comfy-film-editor a{color:var(--film-accent)}
#comfy-film-editor input,#comfy-film-editor textarea,#comfy-film-editor select{box-sizing:border-box;width:100%;padding:10px 12px;background:#12171e;color:#eef2f7;border:1px solid #465263;border-radius:8px;font:inherit}
#comfy-film-editor textarea{resize:vertical;min-height:100px}
#comfy-film-editor input[type=checkbox]{width:auto;margin-right:8px;accent-color:var(--film-accent)}
#comfy-film-editor input[type=file]{font-size:12px}
#comfy-film-editor label{display:block;margin:12px 0;color:#dbe3ec;font-size:13px}
#comfy-film-editor label>input,#comfy-film-editor label>textarea,#comfy-film-editor label>select{margin-top:5px}
#comfy-film-editor button{padding:9px 14px;margin:3px;border:1px solid #465263;border-radius:8px;background:#293341;color:#eef2f7;cursor:pointer;font:inherit;font-weight:550}
#comfy-film-editor button:hover{border-color:var(--film-accent);background:#344354}
#comfy-film-editor button:disabled{opacity:.45;cursor:wait}
#comfy-film-editor :focus-visible{outline:2px solid var(--film-accent);outline-offset:3px}
#comfy-film-editor :invalid{border-color:#ffb2a6}
#comfy-film-editor summary{cursor:pointer;font-weight:600;padding:8px 0}
#comfy-film-editor section,#comfy-film-editor details.comfy-film-shot{border:1px solid var(--film-line);padding:20px;margin:14px 0;border-radius:12px;background:var(--film-panel)}
#comfy-film-editor .film-header{padding:14px 26px 10px;border-bottom:1px solid var(--film-line);display:flex;align-items:center;gap:24px;justify-content:space-between}
#comfy-film-editor .film-header select{max-width:280px}
#comfy-film-editor .film-header p{margin:0}
#comfy-film-editor .film-actions{position:sticky;top:0;z-index:3;padding:10px 22px;background:#14171df5;border-bottom:1px solid var(--film-line);display:flex;gap:4px;flex-wrap:wrap;align-items:center}
#comfy-film-editor .film-primary{background:var(--film-accent);color:#10241c;border-color:var(--film-accent)}
#comfy-film-editor .film-status{padding:8px 26px;margin:0;min-height:36px}
#comfy-film-editor .film-workspace{padding:0 26px 24px}
#comfy-film-editor .film-preview{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(230px,1fr);gap:24px;align-items:center;background:#10141a}
#comfy-film-editor .film-screen{height:clamp(160px,28vh,300px);background:#090c11;border-radius:10px;display:grid;place-items:center;overflow:hidden;min-width:0}
#comfy-film-editor .film-screen video{display:block;width:100%;height:100%;max-height:300px;object-fit:contain}
#comfy-film-editor .film-screen p{max-width:300px;text-align:center;padding:24px}
#comfy-film-editor .film-badge{display:inline-block;border:1px solid #476455;background:#223a30;color:#bcf1d8;border-radius:20px;padding:4px 10px;font-size:12px;margin:0 0 12px}
#comfy-film-editor .film-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}
#comfy-film-editor .film-tabs button[aria-pressed=true]{border-color:var(--film-accent);color:var(--film-accent);background:#24352f}
#comfy-film-editor .film-strip{display:flex;gap:12px;overflow-x:auto;padding:4px 2px 12px;scroll-snap-type:x proximity}
#comfy-film-editor .film-tile{flex:0 0 174px;min-width:0;white-space:normal;text-align:left;padding:8px;background:#111720;scroll-snap-align:start}
#comfy-film-editor .film-tile[aria-pressed=true]{border:2px solid var(--film-accent);padding:7px}
#comfy-film-editor .film-thumb{display:grid;place-items:center;aspect-ratio:16/9;background:linear-gradient(145deg,#324759,#202a35);border-radius:5px;overflow:hidden;margin-bottom:8px;font-size:26px;color:#a7bdcf}
#comfy-film-editor .film-thumb img{width:100%;height:100%;object-fit:cover}
#comfy-film-editor .film-tile strong,#comfy-film-editor .film-tile small{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#comfy-film-editor .film-tile .film-shot-status{white-space:normal;color:#d4e4f2;font-size:11px;margin-top:4px}
#comfy-film-editor .film-frames,#comfy-film-editor .film-reference-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
#comfy-film-editor .film-frames>section{margin:4px 0;padding:14px;background:#141b23}
#comfy-film-editor .film-reference-grid>section{margin:0;background:#141b23}
#comfy-film-editor .film-reference-chips{display:flex;flex-wrap:wrap;gap:8px}
#comfy-film-editor .film-reference-chips label{padding:6px 10px;border:1px solid var(--film-line);border-radius:20px;margin:4px 0}
#comfy-film-editor .film-help{padding:12px 16px;border-left:3px solid #7192b3;background:#182736;border-radius:4px}
#comfy-film-editor .film-sound-lane{position:relative;height:34px;background:#111820;border-radius:6px;margin:10px 0;overflow:hidden}
#comfy-film-editor .film-sound-clip{position:absolute;height:100%;min-width:2px;background:#315c51;border:1px solid #9adebf;border-radius:5px}
#comfy-film-editor .film-tools{padding:0 26px 20px}
#comfy-film-editor .film-review{margin:0 26px 24px}
#comfy-film-editor .film-review video{width:260px;max-width:100%}
#comfy-film-editor table{width:100%;text-align:left;border-collapse:collapse}
#comfy-film-editor td,#comfy-film-editor th{padding:10px;vertical-align:top;border-bottom:1px solid var(--film-line)}
@media(max-width:760px){#comfy-film-editor{width:98vw;height:96vh;max-height:96vh;border-radius:10px}#comfy-film-editor .film-header{display:block;padding:16px}#comfy-film-editor .film-header select{max-width:100%;margin-top:12px}#comfy-film-editor .film-workspace{padding:0 12px 16px}#comfy-film-editor .film-preview,#comfy-film-editor .film-frames,#comfy-film-editor .film-reference-grid{grid-template-columns:1fr}#comfy-film-editor .film-actions{padding:8px}#comfy-film-editor .film-tile{flex-basis:150px}#comfy-film-editor .film-review{margin:0 12px 16px}#comfy-film-editor .film-status{padding:12px}#comfy-film-editor section{padding:14px}}
`
