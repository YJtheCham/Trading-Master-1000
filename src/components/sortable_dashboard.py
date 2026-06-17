"""
可拖拽排序仪表盘 — 分两步: iframe排序 + Streamlit按钮读取
"""
import json, uuid
import streamlit as st
import streamlit.components.v1 as components


def sortable_dashboard(cards: list[dict], height: int = 600) -> list[str] | None:
    dark = cards[0].get("dark", False) if cards else False
    bg = "#0d1117" if dark else "#f8f9fa"
    card_bg = "#161b22" if dark else "#fff"
    fg = "#c9d1d9" if dark else "#212529"
    border = "#30363d" if dark else "#dee2e6"
    store_key = f"_sort_{uuid.uuid4().hex[:6]}"
    cards_js = json.dumps([{"key": c["key"], "name": c["name"],
                            "price": str(c["price"]), "change": c.get("change", 0)}
                           for c in cards])

    html = f"""<!DOCTYPE html><html><head><style>
body {{ margin:0; padding:0; background:{bg}; font-family:-apple-system,sans-serif; }}
.grid {{ display:flex; flex-wrap:wrap; gap:6px; padding:4px; }}
.card {{ background:{card_bg}; color:{fg}; border:1px solid {border}; border-radius:8px;
        padding:8px 6px; width:calc(25% - 12px); min-width:100px;
        box-sizing:border-box; text-align:center; cursor:grab; user-select:none; }}
.card.dragging {{ opacity:0.4; }}
.name {{ font-size:0.7rem; color:#8b949e; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.price {{ font-size:1.1rem; font-weight:700; margin:2px 0; }}
.change {{ font-size:0.75rem; font-weight:500; }}
.btn-bar {{ text-align:center; margin:6px 0; }}
.save-btn {{ padding:6px 20px; background:#1f6feb; color:#fff; border:none; border-radius:6px; cursor:pointer; font-size:0.85rem; }}
</style></head><body>
<div class="btn-bar"><button class="save-btn" onclick="save()">保存排序 (然后点下方 Streamlit 按钮)</button></div>
<div class="grid" id="grid"></div>
<script>
var CARDS = {cards_js};
var grid = document.getElementById("grid");
var dragSrc = null;
function render() {{
  grid.innerHTML = "";
  CARDS.forEach(function(c, idx) {{
    var chg = parseFloat(c.change) || 0;
    var color = chg >= 0 ? "#26a69a" : "#ef5350";
    var div = document.createElement("div");
    div.className = "card"; div.draggable = true; div.dataset.idx = idx;
    div.innerHTML = '<div class="name">' + c.name + '</div>' +
      '<div class="price">' + c.price + '</div>' +
      '<div class="change" style="color:' + color + '">' + (chg>=0?'+':'') + chg.toFixed(2) + '%</div>';
    div.addEventListener("dragstart", function(e) {{ dragSrc = this; this.classList.add("dragging"); }});
    div.addEventListener("dragend", function(e) {{ this.classList.remove("dragging"); }});
    div.addEventListener("dragover", function(e) {{ e.preventDefault(); }});
    div.addEventListener("drop", function(e) {{
      e.preventDefault();
      if (dragSrc !== this) {{
        var from = parseInt(dragSrc.dataset.idx); var to = parseInt(this.dataset.idx);
        CARDS.splice(to, 0, CARDS.splice(from, 1)[0]); render();
      }}
    }});
    grid.appendChild(div);
  }});
}}
function save() {{
  sessionStorage.setItem("{store_key}", JSON.stringify(CARDS.map(function(c){{return c.key;}})));
  alert("已保存! 关闭此提示后, 点下方「读取保存的排序」按钮");
}}
render();
</script></body></html>"""

    components.html(html, height=height, scrolling=True)

    # Step 2: 读取 sessionStorage 的 Streamlit 按钮
    if st.button("📥 读取保存的排序", use_container_width=True, type="primary",
                 help="先在上方排序区点保存, 再点此按钮"):
        read_js = f"""
<script>
var val = sessionStorage.getItem("{store_key}");
var input = window.parent.document.querySelector('input[data-sort-store="true"]');
if (input && val) {{ input.value = val; input.dispatchEvent(new Event('input', {{bubbles:true}})); }}
</script>"""
        st.markdown(read_js, unsafe_allow_html=True)
        # Hidden input to receive the value
        stored = st.text_input("_sort_val", key=f"sort_input_{store_key}",
                               label_visibility="collapsed")
        if stored:
            try:
                return json.loads(stored)
            except (json.JSONDecodeError, TypeError):
                pass
    return None
