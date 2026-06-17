"""
可拖拽排序组件 — HTML5 原生 DnD + sessionStorage 回传
零外部依赖, 全浏览器兼容
"""
import json
import streamlit.components.v1 as components
import streamlit as st


_RECOVERY_DONE = "_sort_recovery_done"


def sortable_cards(cards: list[dict], height: int = 550) -> list[str] | None:
    """
    Args:
        cards: [{"key": "A-600519", "name": "茅台", "price": "1520.50", "change": 1.2}, ...]
    Returns:
        新 key 列表, 或 None
    """
    dark = cards[0].get("dark", False) if cards else False
    bg = "#0d1117" if dark else "#f8f9fa"
    card_bg = "#161b22" if dark else "#fff"
    fg = "#c9d1d9" if dark else "#212529"
    border = "#30363d" if dark else "#dee2e6"
    cards_js = json.dumps(cards, ensure_ascii=False)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body {{ margin:0; padding:8px; background:{bg}; font-family:-apple-system,BlinkMacSystemFont,sans-serif; }}
.grid {{ display:flex; flex-wrap:wrap; gap:8px; }}
.card {{ background:{card_bg}; color:{fg}; border:1px solid {border}; border-radius:10px;
        padding:10px 8px; width:calc(25% - 14px); min-width:120px; box-sizing:border-box;
        text-align:center; cursor:grab; user-select:none; transition:box-shadow .15s; }}
.card:hover {{ box-shadow:0 2px 12px rgba(31,111,235,0.3); }}
.card:active {{ cursor:grabbing; }}
.card.dragging {{ opacity:0.35; }}
.card.drag-over {{ box-shadow:0 0 0 2px #1f6feb !important; }}
.nm {{ font-size:0.72rem; color:#8b949e; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.pr {{ font-size:1.15rem; font-weight:700; margin:4px 0; }}
.ch {{ font-size:0.78rem; font-weight:500; }}
.toolbar {{ text-align:center; margin-top:12px; }}
.btn {{ padding:8px 24px; background:#1f6feb; color:#fff; border:none; border-radius:6px;
        cursor:pointer; font-size:0.85rem; margin:0 4px; }}
.btn-gray {{ background:#484f58; }}
</style></head><body>
<div class="grid" id="g"></div>
<div class="toolbar">
  <button class="btn" onclick="save()">💾 保存</button>
  <button class="btn btn-gray" onclick="cancel()">取消</button>
</div>
<script>
var ITEMS = {cards_js};
var g = document.getElementById("g");
var dragSrc = null;

function render() {{
  g.innerHTML = "";
  ITEMS.forEach(function(c, i) {{
    var chg = parseFloat(c.change) || 0;
    var clr = chg >= 0 ? "#26a69a" : "#ef5350";
    var d = document.createElement("div");
    d.className = "card"; d.draggable = true; d.setAttribute("data-i", i);
    d.innerHTML = '<div class="nm">' + c.name + '</div>' +
      '<div class="pr">' + (c.price||"-") + '</div>' +
      '<div class="ch" style="color:' + clr + '">' + (chg>=0?'+':'') + chg.toFixed(2) + '%</div>';
    d.addEventListener("dragstart", function(e) {{ dragSrc = this; this.classList.add("dragging"); e.dataTransfer.setData("t", this.getAttribute("data-i")); }});
    d.addEventListener("dragend", function(e) {{ this.classList.remove("dragging"); }});
    d.addEventListener("dragover", function(e) {{ e.preventDefault(); this.classList.add("drag-over"); }});
    d.addEventListener("dragleave", function(e) {{ this.classList.remove("drag-over"); }});
    d.addEventListener("drop", function(e) {{
      e.preventDefault(); this.classList.remove("drag-over");
      if (dragSrc !== this) {{
        var f = parseInt(dragSrc.getAttribute("data-i"));
        var t = parseInt(this.getAttribute("data-i"));
        ITEMS.splice(t, 0, ITEMS.splice(f, 1)[0]);
        render();
      }}
    }});
    g.appendChild(d);
  }});
}}
function save() {{
  var keys = ITEMS.map(function(c) {{ return c.key; }});
  sessionStorage.setItem("_sortable_keys", JSON.stringify(keys));
  window.top.location.reload();
}}
function cancel() {{ window.top.location.reload(); }}
render();
</script></body></html>"""

    # === 恢复机制: 检查 sessionStorage ===
    if not st.session_state.get(_RECOVERY_DONE):
        # 注入 JS 检查 sessionStorage, 有值则写入 query param
        st.markdown("""
<script>
(function(){
  var v = sessionStorage.getItem("_sortable_keys");
  if (v) {
    sessionStorage.removeItem("_sortable_keys");
    var sp = new URLSearchParams(window.location.search);
    sp.set("_s", v);
    window.location.search = sp.toString();
  }
})();
</script>""", unsafe_allow_html=True)
        st.session_state[_RECOVERY_DONE] = True
        return None

    # === 渲染组件 ===
    components.html(html, height=height, scrolling=True)

    # === 读取恢复的排序 ===
    raw = st.query_params.get("_s", "")
    if raw:
        st.query_params.pop("_s", None)
        st.session_state[_RECOVERY_DONE] = False
        try:
            return json.loads(raw)
        except Exception:
            pass
    return None
