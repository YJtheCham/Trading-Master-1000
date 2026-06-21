"""
可拖拽 + 可点击 + 可分组 卡片仪表盘

JS↔Python 通信: 通过 Streamlit.setComponentValue 回传动作,
无需页面跳转, 无 query_params 竞态.
"""
import json
import streamlit.components.v1 as components
import streamlit as st


def dash_cards(cards: list[dict], groups: list[str],
               height: int = 500) -> dict | None:
    if not cards:
        return None

    dark = cards[0].get("dark", False) if cards else False
    monitored_keys = [c["key"] for c in cards if c.get("monitored")]
    bg = "#0d1117" if dark else "#ffffff"
    card_bg = "#161b22" if dark else "#f8f9fa"
    fg = "#c9d1d9" if dark else "#212529"
    border = "#30363d" if dark else "#e9ecef"
    border_hover = "#1f6feb" if dark else "#1f77b4"
    muted = "#8b949e" if dark else "#6c757d"
    tag_bg = "#21262d" if dark else "#e8f0fe"
    tag_fg = "#58a6ff" if dark else "#1f77b4"
    modal_bg = "#161b22" if dark else "#fff"
    cards_js = json.dumps(cards, ensure_ascii=False)
    groups_js = json.dumps(groups, ensure_ascii=False)
    mon_js = json.dumps(monitored_keys, ensure_ascii=False)

    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{BG;font-family:-apple-system,BlinkMacSystemFont,sans-serif;padding:6px;-webkit-user-select:none;user-select:none}
.grid{display:flex;flex-wrap:wrap;gap:8px}
.card{background:CBG;color:FG;border:1px solid BR;border-radius:10px;padding:10px 8px 6px;width:calc(25% - 8px);min-width:140px;text-align:center;cursor:pointer;transition:border-color .15s,box-shadow .15s,transform .1s;position:relative}
.card:hover{border-color:BH;box-shadow:0 2px 8px rgba(0,0,0,.12)}
.card.monitored{border-color:#22c55e;box-shadow:0 0 0 1px #22c55e inset}
.card.dragging{opacity:.4;transform:scale(.95)}
.card.drag-over{border-color:BH!important;box-shadow:0 0 0 2px BH!important}
.tag{display:inline-block;font-size:.62rem;padding:1px 6px;border-radius:8px;background:TBG;color:TFG;cursor:pointer;margin-top:4px;border:none;line-height:1.5}
.tag:hover{filter:brightness(1.15)}
.date{font-size:.68rem;color:MU;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:2px}
.name{font-size:.78rem;color:MU;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.price{font-size:1.2rem;font-weight:700;margin:3px 0}
.chg{font-size:.8rem;font-weight:500}
.mo{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.45);z-index:100;justify-content:center;align-items:center}
.mo.on{display:flex}
.mb{background:MOBG;border-radius:12px;padding:16px;min-width:200px;max-width:280px;box-shadow:0 8px 32px rgba(0,0,0,.3)}
.mb h4{font-size:.85rem;margin-bottom:10px;color:FG}
.mi{padding:6px 10px;border-radius:6px;cursor:pointer;font-size:.78rem;color:FG;margin-bottom:2px}
.mi:hover{background:TBG}
.mi.sel{color:TFG;font-weight:600}
.mn{display:flex;gap:4px;margin-top:8px}
.mn input{flex:1;padding:4px 8px;border-radius:6px;border:1px solid BR;background:BG;color:FG;font-size:.78rem;outline:none}
.mn button{padding:4px 10px;border-radius:6px;background:TBG;color:TFG;border:none;cursor:pointer;font-size:.75rem}
@media(max-width:768px){.card{width:calc(50% - 6px);min-width:0;padding:8px 6px}.price{font-size:1rem}.name,.date{font-size:.68rem}}
</style></head><body>
<div class="grid" id="g"></div>
<div class="mo" id="ov"><div class="mb" id="mb"></div></div>
<script>
var I=ITEMS,G=GRPS,M=MON,dS=null,drg=false,sX=0,sY=0;
function el(t,c){var e=document.createElement(t);if(c)e.className=c;return e}
function tx(t){return document.createTextNode(t)}
function send(v){window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setComponentValue',value:v},'*')}
function rn(){
  g.innerHTML="";
  I.forEach(function(c,i){
    var ch=+c.change||0,cl=ch>=0?"#26a69a":"#ef5350",m=M.indexOf(c.key)>=0;
    var d=el("div","card"+(m?" monitored":""));
    d.draggable=true;d.dataset.idx=i;
    var de=el("div","date");de.appendChild(tx(c.date_label||""));d.appendChild(de);
    var nm=el("div","name");nm.appendChild(tx(c.name));d.appendChild(nm);
    var pr=el("div","price");pr.appendChild(tx(c.price||"-"));d.appendChild(pr);
    var cg=el("div","chg");cg.style.color=cl;cg.appendChild(tx((ch>=0?"+":"")+ch.toFixed(2)+"%"));d.appendChild(cg);
    var tg=el("button","tag");tg.dataset.gk=c.key;tg.appendChild(tx(c.group||"默认"));d.appendChild(tg);
    d.addEventListener("dragstart",function(e){dS=this;drg=false;sX=e.clientX;sY=e.clientY;this.classList.add("dragging");e.dataTransfer.setData("t","x")});
    d.addEventListener("dragend",function(e){this.classList.remove("dragging");if(!drg&&Math.abs(e.clientX-sX)<5&&Math.abs(e.clientY-sY)<5)send({action:"click",key:c.key})});
    d.addEventListener("dragover",function(e){e.preventDefault();this.classList.add("drag-over")});
    d.addEventListener("dragleave",function(){this.classList.remove("drag-over")});
    d.addEventListener("drop",function(e){e.preventDefault();this.classList.remove("drag-over");if(dS&&dS!==this){drg=true;var f=+dS.dataset.idx,t=+this.dataset.idx;I.splice(t,0,I.splice(f,1)[0]);rn();send({action:"reorder",keys:I.map(function(cc){return cc.key})})}});
    g.appendChild(d)
  });
  g.querySelectorAll(".tag").forEach(function(b){b.addEventListener("click",function(e){e.stopPropagation();gm(this.dataset.gk)})})
}
function gm(k){
  var it=I.find(function(c){return c.key===k});if(!it)return;
  var cu=it.group||"默认",h="<h4>修改分组: "+it.name+"</h4>";
  G.forEach(function(grp){var a=grp===cu?" sel":"";h+='<div class="mi'+a+'" data-mk="'+k+'" data-mg="'+grp+'">'+grp+"</div>"});
  h+='<div class="mn"><input id="ni" placeholder="新分组名..."><button id="nb">+</button></div>';
  mb.innerHTML=h;ov.classList.add("on");
  mb.querySelectorAll(".mi").forEach(function(el){el.addEventListener("click",function(){sg(this.dataset.mk,this.dataset.mg)})});
  var ni=document.getElementById("ni"),nb=document.getElementById("nb");
  if(ni)ni.addEventListener("keydown",function(e){if(e.key==="Enter")ng(k)});
  if(nb)nb.addEventListener("click",function(){ng(k)});
}
ov.addEventListener("click",function(e){if(e.target===ov)ov.classList.remove("on")});
function sg(k,grp){ov.classList.remove("on");send({action:"group",key:k,group:grp})}
function ng(k){var v=document.getElementById("ni");if(!v||!v.value.trim())return;ov.classList.remove("on");send({action:"new_group",key:k,name:v.value.trim()})}
rn();
</script></body></html>"""

    replacements = [
        ("MOBG", modal_bg), ("ITEMS", cards_js), ("GRPS", groups_js), ("MON", mon_js),
        ("CBG", card_bg), ("TBG", tag_bg), ("TFG", tag_fg),
        ("BH", border_hover), ("BR", border), ("BG", bg), ("FG", fg), ("MU", muted),
    ]
    for placeholder, value in replacements:
        html = html.replace(placeholder, value)

    result = components.html(html, height=height, scrolling=True)

    if isinstance(result, dict):
        action = result.get("action")
        if action in ("click", "reorder", "group", "new_group"):
            return result

    return None
