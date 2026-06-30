"""
分组拖拽排序组件 - 独立组件文件, 与 dash_cards 同模式
"""
import json
import streamlit.components.v1 as components


def group_drag(groups_data: list[dict], dark: bool = False) -> list[str] | None:
    """拖拽排序分组列表, 返回新的分组名顺序"""
    if not groups_data:
        return None

    bg = "#0d1117" if dark else "#fff"
    fg = "#c9d1d9" if dark else "#212529"
    mu = "#8b949e" if dark else "#6c757d"
    br_c = "#30363d" if dark else "#e9ecef"
    ac = "#1f6feb" if dark else "#1f77b4"
    groups_js = json.dumps(groups_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:{bg};font-family:-apple-system,sans-serif;padding:4px}}
.it{{display:flex;align-items:center;padding:5px 10px;margin:2px 0;border:1px solid {br_c};border-radius:6px;cursor:grab;background:{bg};color:{fg};font-size:.78rem;user-select:none}}
.it:hover{{background:{ac}18}}
.it:active{{cursor:grabbing;background:{ac}25}}
.it.ov{{border-color:{ac};background:{ac}22}}
.it.dragging{{opacity:.3}}
.hd{{margin-right:8px;color:{mu};font-size:.65rem}}
.nm{{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.ct{{color:{mu};font-size:.68rem;margin-left:8px;flex-shrink:0}}
</style></head><body>
<div id="glist"></div>
<script>
var G=GROUPS;
function send(v){{window.parent.postMessage({{isStreamlitMessage:true,type:'streamlit:setComponentValue',value:v}},'*');console.log('group_drag send',v)}}
function rn(){{
 var gl=document.getElementById("glist"),h="";
 G.forEach(function(g,i){{h+='<div class=\"it\" draggable=\"true\" data-idx=\"'+i+'\"><span class=\"hd\">⠿</span><span class=\"nm\">'+g.name+'</span><span class=\"ct\">'+g.count+'只</span></div>'}})
 gl.innerHTML=h;
 gl.querySelectorAll(".it").forEach(function(d){{
  d.ondragstart=function(e){{d.classList.add("dragging");e.dataTransfer.effectAllowed="move";e.dataTransfer.setData("text/idx",this.dataset.idx)}};
  d.ondragend=function(e){{d.classList.remove("dragging");gl.querySelectorAll(".it").forEach(function(x){{x.classList.remove("ov")}})}};
  d.ondragover=function(e){{e.preventDefault();e.dataTransfer.dropEffect="move";this.classList.add("ov")}};
  d.ondragleave=function(){{this.classList.remove("ov")}};
  d.ondrop=function(e){{
   e.preventDefault();e.stopPropagation();this.classList.remove("ov");
   var f=parseInt(e.dataTransfer.getData("text/idx"),10),t=parseInt(this.dataset.idx,10);
   if(!isNaN(f)&&!isNaN(t)&&f!==t){{
    var item=G.splice(f,1)[0];G.splice(t,0,item);
    rn();send({{action:"reorder",groups:G.map(function(x){{return x.name}})}})
   }}
  }}
 }})
}}
rn();
</script></body></html>"""
    html = html.replace("GROUPS", groups_js)

    result = components.html(html, height=max(80, len(groups_data) * 40 + 16), scrolling=True)

    if isinstance(result, dict) and result.get("action") == "reorder":
        return result.get("groups", [])

    return None
