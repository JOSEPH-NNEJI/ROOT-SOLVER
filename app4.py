import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  SYMBOLIC PARSER
# ─────────────────────────────────────────────
class FunctionParser:
    def __init__(self, func_str):
        self.func_str = func_str
        self.valid = True
        self.error_msg = ""
        self.x = sp.Symbol('x')
        try:
            from sympy.parsing.sympy_parser import (
                parse_expr, standard_transformations,
                implicit_multiplication_application, convert_xor
            )
            tf = standard_transformations + (implicit_multiplication_application, convert_xor)
            self.expr   = parse_expr(func_str, transformations=tf)
            self.d_expr = sp.diff(self.expr, self.x)
            self.d2_expr= sp.diff(self.d_expr, self.x)
            self.f_num  = sp.lambdify(self.x, self.expr,   "numpy")
            self.df_num = sp.lambdify(self.x, self.d_expr, "numpy")
        except Exception as e:
            self.valid = False
            self.error_msg = f"Invalid function: {e}"

    def f(self, v):
        try:
            result = float(self.f_num(v))
            return result if np.isfinite(result) else None
        except Exception:
            return None

    def df(self, v):
        try:
            result = float(self.df_num(v))
            return result if np.isfinite(result) else None
        except Exception:
            return None

# ─────────────────────────────────────────────
#  METHOD SOLVERS
# ─────────────────────────────────────────────

def solve_bisection(parser, a, b, tol, max_iter=100):
    fa = parser.f(a)
    fb = parser.f(b)
    if fa is None or fb is None:
        return None, "Cannot evaluate function at bounds.", []
    if fa * fb > 0:
        return None, "f(a) and f(b) must have opposite signs for Bisection.", []

    history = []
    curr_a, curr_b = float(a), float(b)
    fa = parser.f(curr_a)

    for i in range(max_iter):
        xm = (curr_a + curr_b) / 2.0
        fxm = parser.f(xm)
        if fxm is None:
            return xm, "Function undefined at midpoint.", history

        error = abs(curr_b - curr_a) / 2.0
        history.append({
            "Iter": i + 1,
            "a": round(curr_a, 8),
            "b": round(curr_b, 8),
            "xₘ (midpoint)": round(xm, 8),
            "f(xₘ)": f"{fxm:.6e}",
            "|b - a| / 2": f"{error:.6e}",
            "Decision": "Root in [a, xₘ]" if fa * fxm < 0 else "Root in [xₘ, b]"
        })

        if error < tol or abs(fxm) < tol:
            return xm, None, history

        if fa * fxm < 0:
            curr_b = xm
        else:
            curr_a = xm
            fa = fxm

    return xm, None, history


def solve_newton_raphson(parser, x0, tol, max_iter=100):
    history = []
    xn = float(x0)

    for i in range(max_iter):
        fxn  = parser.f(xn)
        dfxn = parser.df(xn)

        if fxn is None:
            return xn, "Function undefined at current point.", history
        if dfxn is None or abs(dfxn) < 1e-14:
            return xn, "Derivative is zero or undefined — Newton-Raphson failed (flat spot / singularity).", history

        x_new = xn - fxn / dfxn
        error  = abs(x_new - xn)

        history.append({
            "Iter": i + 1,
            "xₙ": round(xn, 8),
            "f(xₙ)": f"{fxn:.6e}",
            "f′(xₙ)": f"{dfxn:.6e}",
            "xₙ₊₁ = xₙ - f/f′": round(x_new, 8),
            "|xₙ₊₁ - xₙ|": f"{error:.6e}",
        })

        if error < tol:
            return x_new, None, history
        xn = x_new

    return xn, "Max iterations reached — may not have converged.", history


def solve_secant(parser, x0, x1, tol, max_iter=100):
    history = []
    xn_1 = float(x0)
    xn   = float(x1)

    for i in range(max_iter):
        fxn_1 = parser.f(xn_1)
        fxn   = parser.f(xn)

        if fxn is None or fxn_1 is None:
            return xn, "Function undefined.", history

        denom = fxn - fxn_1
        if abs(denom) < 1e-14:
            return xn, "Secant denominator too small — method failed.", history

        x_new = xn - fxn * (xn - xn_1) / denom
        error  = abs(x_new - xn)

        history.append({
            "Iter": i + 1,
            "xₙ₋₁": round(xn_1, 8),
            "xₙ": round(xn, 8),
            "f(xₙ₋₁)": f"{fxn_1:.6e}",
            "f(xₙ)": f"{fxn:.6e}",
            "xₙ₊₁": round(x_new, 8),
            "|xₙ₊₁ - xₙ|": f"{error:.6e}",
        })

        if error < tol:
            return x_new, None, history

        xn_1 = xn
        xn   = x_new

    return xn, "Max iterations reached.", history


def solve_false_position(parser, a, b, tol, max_iter=100):
    fa = parser.f(a)
    fb = parser.f(b)
    if fa is None or fb is None:
        return None, "Cannot evaluate function at bounds.", []
    if fa * fb > 0:
        return None, "f(a) and f(b) must have opposite signs for False Position.", []

    history = []
    curr_a, curr_b = float(a), float(b)
    fa = parser.f(curr_a)
    fb = parser.f(curr_b)

    for i in range(max_iter):
        # False position / regula falsi formula
        denom = fb - fa
        if abs(denom) < 1e-14:
            return curr_a, "Denominator too small.", history

        xr = curr_b - fb * (curr_b - curr_a) / denom
        fxr = parser.f(xr)
        if fxr is None:
            return xr, "Function undefined at estimate.", history

        error = abs(fxr)
        history.append({
            "Iter": i + 1,
            "a": round(curr_a, 8),
            "b": round(curr_b, 8),
            "xᵣ (false pos.)": round(xr, 8),
            "f(xᵣ)": f"{fxr:.6e}",
            "|f(xᵣ)|": f"{error:.6e}",
            "Update": "b ← xᵣ" if fa * fxr < 0 else "a ← xᵣ"
        })

        if error < tol:
            return xr, None, history

        if fa * fxr < 0:
            curr_b = xr
            fb = fxr
        else:
            curr_a = xr
            fa = fxr

    return xr, None, history


def solve_hybrid(parser, a, b, tol, max_iter=100):
    fa = parser.f(a)
    fb = parser.f(b)
    if fa is None or fb is None:
        return None, "Cannot evaluate function at bounds.", []
    if fa * fb > 0:
        return None, "f(a) and f(b) must have opposite signs for Hybrid method.", []

    history = []
    curr_a, curr_b = float(a), float(b)
    xn = (curr_a + curr_b) / 2.0
    fa = parser.f(curr_a)

    for i in range(max_iter):
        fxn  = parser.f(xn)
        dfxn = parser.df(xn)

        if fxn is None:
            return xn, "Function undefined.", history

        error  = abs(fxn)
        method = "Bisection"
        reason = "Default bracketing step"

        if dfxn is not None and abs(dfxn) > 1e-10:
            x_newton = xn - fxn / dfxn
            if curr_a < x_newton < curr_b:
                xn     = x_newton
                method = "Newton-Raphson"
                reason = "Newton step in-bounds ✓"
            else:
                reason = "Newton step out-of-bounds → fallback"
        else:
            reason = "Derivative ≈ 0 (singularity) → fallback"

        history.append({
            "Iter": i + 1,
            "xₙ (estimate)": round(xn, 8),
            "f(xₙ)": f"{fxn:.6e}",
            "|f(xₙ)| error": f"{error:.6e}",
            "Method used": method,
            "Decision logic": reason,
        })

        if error < tol:
            return xn, None, history

        if method == "Bisection":
            if fa * fxn < 0:
                curr_b = xn
            else:
                curr_a = xn
                fa = fxn
            xn = (curr_a + curr_b) / 2.0

    return xn, None, history


# ─────────────────────────────────────────────
#  CONVERGENCE PLOT HELPER
# ─────────────────────────────────────────────
def get_error_series(history, method_name):
    errors = []
    for row in history:
        for key, val in row.items():
            if "error" in key.lower() or "|f(" in key or "|xₙ₊₁" in key:
                try:
                    errors.append(float(val))
                    break
                except Exception:
                    pass
    return errors


# ─────────────────────────────────────────────
#  STREAMLIT APP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Root-Finding Suite",
    page_icon="📐",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .method-header {
        font-size: 1.3rem; font-weight: 700;
        padding: 0.4rem 0; margin-bottom: 0.5rem;
        border-bottom: 2px solid #4CAF50;
    }
    .metric-box {
        background: #f8f9fa; border-radius: 8px;
        padding: 12px 16px; text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-val { font-size: 1.4rem; font-weight: 700; color: #2196F3; }
    .metric-lbl { font-size: 0.75rem; color: #666; margin-top: 2px; }
    .winner-badge {
        background: #4CAF50; color: white;
        border-radius: 20px; padding: 2px 12px;
        font-size: 0.75rem; font-weight: 700;
    }
    .fail-badge {
        background: #f44336; color: white;
        border-radius: 20px; padding: 2px 12px;
        font-size: 0.75rem; font-weight: 700;
    }
    .step-box {
        background: #f0f7ff; border-left: 4px solid #2196F3;
        padding: 10px 14px; border-radius: 0 6px 6px 0;
        margin-bottom: 8px; font-family: monospace; font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Title ──
st.title("📐 Root-Finding Methods Suite")
st.markdown(
    "**Covenant University Final Year Project** — Compare Bisection, Newton-Raphson, "
    "Secant, False Position, and the Hybrid solver. View step-by-step solutions and benchmarks."
)
st.divider()

# ─────────────────────────────────────────────
#  SIDEBAR — Shared inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Problem Setup")

    st.subheader("Preset Test Cases")
    presets = {
        "Custom": ("", -2.0, 2.0, 1.0),
        "Case 1 — Sanity Check: x²-2":       ("x**2 - 2",        1.0,  2.0,  1.5),
        "Case 2 — Bad Guess Loop: x³-2x+2":  ("x**3 - 2*x + 2",  -2.0, 1.0,  0.0),
        "Case 3 — Singularity: x^(1/3)":     ("x**(1/3)",         -1.0, 1.0,  0.5),
        "Case 4 — Flat Spot: arctan(x)":     ("atan(x)",          -3.0, 3.0,  2.5),
        "Case 5 — Transcendental: e^-x - sin(x)": ("exp(-x) - sin(x)", 0.0, 1.0, 0.5),
        "Case 6 — Stiff Poly: x¹⁰-1":       ("x**10 - 1",        0.0,  1.3,  0.5),
    }
    chosen = st.selectbox("Load preset", list(presets.keys()))
    p_func, p_a, p_b, p_x0 = presets[chosen]

    func_input = st.text_input("f(x) =", value=p_func or "x**3 - 2*x + 2",
                                help="Python syntax: x**2, exp(x), sin(x), log(x)")
    st.caption("Use: `x**2`, `exp(x)`, `sin(x)`, `cos(x)`, `log(x)`, `atan(x)`, `sqrt(x)`")

    st.subheader("Interval [a, b]")
    col_a, col_b = st.columns(2)
    a_val = col_a.number_input("a", value=float(p_a))
    b_val = col_b.number_input("b", value=float(p_b))

    st.subheader("Initial Guess x₀")
    x0_val = st.number_input("x₀ (for Newton & Secant)", value=float(p_x0))

    st.subheader("Second Point x₁")
    x1_val = st.number_input("x₁ (Secant only)", value=float(p_b))

    st.subheader("Parameters")
    tol      = st.number_input("Tolerance ε", value=1e-6, format="%.1e")
    max_iter = st.slider("Max iterations", 10, 200, 100)

    st.subheader("Display options")
    show_steps   = st.checkbox("Show step-by-step tables", value=True)
    show_graph   = st.checkbox("Show convergence graphs",  value=True)
    show_compare = st.checkbox("Show comparison dashboard", value=True)

    run = st.button("🚀 Solve All Methods", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
#  MAIN LOGIC
# ─────────────────────────────────────────────
if run:
    parser = FunctionParser(func_input)
    if not parser.valid:
        st.error(f"❌ {parser.error_msg}")
        st.stop()

    # Run all methods and time them
    results = {}

    def run_method(name, fn, *args):
        t0 = time.perf_counter()
        root, err, hist = fn(*args)
        elapsed = time.perf_counter() - t0
        return {
            "root": root,
            "error": err,
            "history": hist,
            "iterations": len(hist),
            "converged": err is None and root is not None,
            "time_ms": elapsed * 1000,
        }

    results["Bisection"]       = run_method("Bisection",       solve_bisection,      parser, a_val, b_val, tol, max_iter)
    results["Newton-Raphson"]  = run_method("Newton-Raphson",  solve_newton_raphson, parser, x0_val, tol, max_iter)
    results["Secant"]          = run_method("Secant",          solve_secant,         parser, x0_val, x1_val, tol, max_iter)
    results["False Position"]  = run_method("False Position",  solve_false_position, parser, a_val, b_val, tol, max_iter)
    results["Hybrid"]          = run_method("Hybrid",          solve_hybrid,         parser, a_val, b_val, tol, max_iter)

    # ── TABS ──
    tabs = st.tabs([
        "🔵 Bisection",
        "🟠 Newton-Raphson",
        "🟣 Secant",
        "🟤 False Position",
        "🟢 Hybrid",
        "📊 Comparison"
    ])

    METHOD_COLORS = {
        "Bisection":      "#2196F3",
        "Newton-Raphson": "#FF9800",
        "Secant":         "#9C27B0",
        "False Position": "#795548",
        "Hybrid":         "#4CAF50",
    }

    METHOD_DESCRIPTIONS = {
        "Bisection": (
            "The Bisection Method is a bracketing technique grounded in the Intermediate Value Theorem. "
            "It repeatedly halves the interval [a, b] where the sign of f changes, guaranteeing convergence "
            "at a linear rate. It is the most reliable method but requires the most iterations."
        ),
        "Newton-Raphson": (
            "Newton-Raphson is an open method that uses the tangent line to the curve at the current estimate. "
            "It converges quadratically (very fast) when the initial guess is close to the root and the derivative "
            "is well-behaved, but it can diverge or fail at flat spots and singularities."
        ),
        "Secant": (
            "The Secant Method approximates the derivative using two previous points, avoiding the need for symbolic "
            "differentiation. It converges superlinearly (order ≈ 1.618) and works well when a derivative is hard to compute."
        ),
        "False Position": (
            "False Position (Regula Falsi) combines the bracketing security of Bisection with a smarter estimate "
            "using a linear interpolation between f(a) and f(b), similar to the Secant formula but keeping the bracket intact."
        ),
        "Hybrid": (
            "The Hybrid Method combines the guaranteed convergence of Bisection with the speed of Newton-Raphson. "
            "At each step, it attempts a Newton step and accepts it only if it stays within the current bracket. "
            "If not, it falls back to a Bisection step. This is the core contribution of this project."
        ),
    }

    STEP_EXPLANATIONS = {
        "Bisection": [
            "Check: f(a)·f(b) < 0 (root is bracketed)",
            "Compute midpoint: xₘ = (a + b) / 2",
            "Evaluate f(xₘ)",
            "If f(a)·f(xₘ) < 0: root in [a, xₘ] → set b = xₘ",
            "Else: root in [xₘ, b] → set a = xₘ",
            "Repeat until |b - a|/2 < ε",
        ],
        "Newton-Raphson": [
            "Start with initial guess x₀",
            "Compute f(xₙ) and f′(xₙ) symbolically",
            "Apply formula: xₙ₊₁ = xₙ - f(xₙ)/f′(xₙ)",
            "Compute error: |xₙ₊₁ - xₙ|",
            "If error < ε: stop. Else: set xₙ = xₙ₊₁ and repeat",
        ],
        "Secant": [
            "Start with two points x₀ and x₁",
            "Compute f(x₀) and f(x₁)",
            "Apply: xₙ₊₁ = xₙ - f(xₙ)·(xₙ - xₙ₋₁) / (f(xₙ) - f(xₙ₋₁))",
            "Shift: xₙ₋₁ ← xₙ, xₙ ← xₙ₊₁",
            "Repeat until |xₙ₊₁ - xₙ| < ε",
        ],
        "False Position": [
            "Check: f(a)·f(b) < 0 (root is bracketed)",
            "Compute: xᵣ = b - f(b)·(b - a) / (f(b) - f(a))",
            "Evaluate f(xᵣ)",
            "If f(a)·f(xᵣ) < 0: root in [a, xᵣ] → b = xᵣ",
            "Else: a = xᵣ",
            "Repeat until |f(xᵣ)| < ε",
        ],
        "Hybrid": [
            "Check: f(a)·f(b) < 0 (root is bracketed)",
            "Start: xₙ = midpoint of [a, b]",
            "Attempt Newton step: x_new = xₙ - f(xₙ)/f′(xₙ)",
            "Safety check: Is x_new ∈ (a, b)?",
            "YES → accept Newton step (fast convergence)",
            "NO or f′≈0 → fall back to Bisection (safe step)",
            "Update bracket [a, b] to keep root trapped",
            "Repeat until |f(xₙ)| < ε",
        ],
    }

    def render_method_tab(tab, method_name, res):
        with tab:
            color = METHOD_COLORS[method_name]
            st.markdown(f"<div class='method-header' style='border-color:{color};'>{method_name}</div>",
                        unsafe_allow_html=True)

            st.info(METHOD_DESCRIPTIONS[method_name])

            # ── How it works ──
            with st.expander("📖 How this method works — step by step", expanded=False):
                for i, step in enumerate(STEP_EXPLANATIONS[method_name], 1):
                    st.markdown(f"<div class='step-box'><b>Step {i}:</b> {step}</div>",
                                unsafe_allow_html=True)

            # ── Result summary ──
            c1, c2, c3, c4 = st.columns(4)
            if res["converged"] and res["root"] is not None:
                c1.success(f"✅ Root ≈ **{res['root']:.10f}**")
            elif res["error"]:
                c1.error(f"❌ {res['error']}")
            else:
                c1.warning(f"⚠️ Root ≈ {res['root']:.10f} (check convergence)")

            c2.metric("Iterations", res["iterations"])
            c3.metric("Time (ms)", f"{res['time_ms']:.3f}")
            c4.metric("Status", "Converged ✓" if res["converged"] else "Failed ✗")

            if res["history"] and show_steps:
                st.subheader("📋 Step-by-Step Iteration Table")

                df = pd.DataFrame(res["history"])

                # colour Decision Logic column for Hybrid
                if method_name == "Hybrid":
                    def style_decision(val):
                        if "Newton" in str(val) and "fallback" not in str(val):
                            return "color: #4CAF50; font-weight: bold;"
                        elif "fallback" in str(val) or "singularity" in str(val):
                            return "color: #f44336; font-weight: bold;"
                        return ""
                    st.dataframe(
                        df.style.map(style_decision, subset=["Decision logic"]),
                        use_container_width=True
                    )
                else:
                    st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode()
                st.download_button(
                    f"⬇️ Download {method_name} iteration table",
                    csv, f"{method_name.replace(' ','_')}_iterations.csv", "text/csv"
                )

            if res["history"] and show_graph:
                st.subheader("📈 Convergence Graph")
                errors = get_error_series(res["history"], method_name)
                if errors:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                    # Left: function curve
                    try:
                        margin = abs(b_val - a_val) * 0.3
                        x_plot = np.linspace(a_val - margin, b_val + margin, 500)
                        y_plot = np.array([parser.f(xi) for xi in x_plot])
                        valid  = np.isfinite(y_plot.astype(float))
                        axes[0].plot(x_plot[valid], y_plot[valid], color=color, lw=2, label="f(x)", alpha=0.8)
                        axes[0].axhline(0, color='black', lw=0.8, ls='--')
                        axes[0].axvline(res["root"], color='red', lw=1.5, ls=':', label=f"root ≈ {res['root']:.6f}")
                        # Plot iteration points
                        for row in res["history"][:20]:
                            for k, v in row.items():
                                if "estimate" in k.lower() or "xₙ (estimate)" in k or "xₘ" in k or "xᵣ" in k or "xₙ₊₁" in k:
                                    try:
                                        xi = float(v)
                                        yi = parser.f(xi)
                                        if yi is not None:
                                            axes[0].scatter(xi, yi, color=color, s=20, alpha=0.6, zorder=5)
                                    except Exception:
                                        pass
                                    break
                        axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
                        axes[0].set_title(f"{method_name} — Function curve")
                        axes[0].legend(); axes[0].grid(True, alpha=0.3)
                        ylim = np.nanpercentile(y_plot[valid].astype(float), [2, 98])
                        axes[0].set_ylim(ylim[0] - 0.5, ylim[1] + 0.5)
                    except Exception:
                        axes[0].text(0.5, 0.5, "Could not plot curve", ha='center', va='center')

                    # Right: error convergence
                    axes[1].semilogy(range(1, len(errors)+1), errors, color=color,
                                     marker='o', ms=4, lw=2, label="Error")
                    axes[1].axhline(tol, color='red', ls='--', label=f"Tolerance = {tol:.0e}")
                    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Error (log scale)")
                    axes[1].set_title(f"{method_name} — Error convergence")
                    axes[1].legend(); axes[1].grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

    # Render individual method tabs
    method_keys = ["Bisection", "Newton-Raphson", "Secant", "False Position", "Hybrid"]
    for tab, key in zip(tabs[:5], method_keys):
        render_method_tab(tab, key, results[key])

    # ─────────────────────────────────────────
    #  COMPARISON TAB
    # ─────────────────────────────────────────
    with tabs[5]:
        st.markdown("<div class='method-header'>📊 Methods Comparison Dashboard</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "This section benchmarks all five methods against each other on the same problem. "
            "The best method for each metric is highlighted in green."
        )

        # ── Summary table ──
        rows = []
        for name, res in results.items():
            root_str = f"{res['root']:.8f}" if res["root"] is not None else "N/A"
            rows.append({
                "Method":     name,
                "Root found": root_str,
                "Iterations": res["iterations"] if res["converged"] else "FAILED",
                "Time (ms)":  f"{res['time_ms']:.4f}",
                "Converged":  "✅ Yes" if res["converged"] else "❌ No",
                "Notes":      res["error"] if res["error"] else "—"
            })

        df_compare = pd.DataFrame(rows)
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        # ── Winner badges ──
        converged_methods = {k: v for k, v in results.items() if v["converged"]}
        if converged_methods:
            fastest_iter = min(converged_methods, key=lambda k: converged_methods[k]["iterations"])
            fastest_time = min(converged_methods, key=lambda k: converged_methods[k]["time_ms"])

            col1, col2, col3 = st.columns(3)
            col1.success(f"⚡ Fewest iterations: **{fastest_iter}** ({converged_methods[fastest_iter]['iterations']} iters)")
            col2.success(f"🏎️ Fastest runtime:  **{fastest_time}** ({converged_methods[fastest_time]['time_ms']:.4f} ms)")
            col3.info(f"🛡️ Most robust: **Hybrid** (guaranteed bracketing + speed)")

        if show_compare:
            st.subheader("📊 Visual Benchmarks")

            fig = plt.figure(figsize=(16, 10))
            gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

            colors_list = [METHOD_COLORS[m] for m in method_keys]
            labels = method_keys

            # ── Bar: Iterations ──
            ax1 = fig.add_subplot(gs[0, 0])
            iters = []
            for m in method_keys:
                r = results[m]
                iters.append(r["iterations"] if r["converged"] else 0)
            bars = ax1.bar(labels, iters, color=colors_list, edgecolor='white', linewidth=0.5)
            best_i = iters.index(min([x for x in iters if x > 0], default=0))
            bars[best_i].set_edgecolor('gold'); bars[best_i].set_linewidth(3)
            ax1.set_title("Iterations to convergence\n(lower = better, gold = winner)", fontsize=11)
            ax1.set_ylabel("Iterations")
            for bar, val, m in zip(bars, iters, method_keys):
                label = str(val) if results[m]["converged"] else "FAIL"
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         label, ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax1.tick_params(axis='x', rotation=20)
            ax1.grid(axis='y', alpha=0.3)

            # ── Bar: Time ──
            ax2 = fig.add_subplot(gs[0, 1])
            times = [results[m]["time_ms"] for m in method_keys]
            bars2 = ax2.bar(labels, times, color=colors_list, edgecolor='white', linewidth=0.5)
            best_t = times.index(min(times))
            bars2[best_t].set_edgecolor('gold'); bars2[best_t].set_linewidth(3)
            ax2.set_title("Runtime (ms)\n(lower = better, gold = winner)", fontsize=11)
            ax2.set_ylabel("Time (ms)")
            for bar, val in zip(bars2, times):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f"{val:.3f}", ha='center', va='bottom', fontsize=8)
            ax2.tick_params(axis='x', rotation=20)
            ax2.grid(axis='y', alpha=0.3)

            # ── Multi-line: Error convergence ──
            ax3 = fig.add_subplot(gs[1, :])
            plotted = False
            for m, color in zip(method_keys, colors_list):
                res = results[m]
                if res["history"] and res["converged"]:
                    errors = get_error_series(res["history"], m)
                    if errors:
                        ax3.semilogy(range(1, len(errors)+1), errors,
                                     color=color, marker='o', ms=3, lw=2,
                                     label=f"{m} ({len(errors)} iters)", alpha=0.85)
                        plotted = True
            if plotted:
                ax3.axhline(tol, color='black', ls='--', lw=1.5, label=f"Tolerance {tol:.0e}")
                ax3.set_xlabel("Iteration number")
                ax3.set_ylabel("Error (log scale)")
                ax3.set_title("Error convergence comparison — all methods", fontsize=12)
                ax3.legend(loc='upper right')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, "No convergence data available", ha='center', va='center',
                         transform=ax3.transAxes, fontsize=14)

            st.pyplot(fig)
            plt.close(fig)

            # ── Hybrid switching pie ──
            hybrid_hist = results["Hybrid"]["history"]
            if hybrid_hist:
                nr_count  = sum(1 for r in hybrid_hist if "Newton" in r.get("Method used","") and "fallback" not in r.get("Decision logic",""))
                bis_count = sum(1 for r in hybrid_hist if "Bisection" in r.get("Method used",""))
                if nr_count + bis_count > 0:
                    st.subheader("🔀 Hybrid Method — Switching breakdown")
                    fig2, ax = plt.subplots(figsize=(5, 4))
                    wedges, texts, autotexts = ax.pie(
                        [nr_count, bis_count],
                        labels=["Newton-Raphson steps", "Bisection steps"],
                        colors=[METHOD_COLORS["Newton-Raphson"], METHOD_COLORS["Bisection"]],
                        autopct='%1.1f%%', startangle=90,
                        wedgeprops=dict(edgecolor='white', linewidth=2)
                    )
                    ax.set_title(f"Hybrid switching breakdown\n({nr_count} NR + {bis_count} Bisection = {nr_count+bis_count} total)")
                    col_l, col_r = st.columns([1, 2])
                    col_l.pyplot(fig2)
                    plt.close(fig2)
                    with col_r:
                        st.markdown(f"""
**Hybrid method analysis:**
- Total iterations: **{nr_count + bis_count}**
- Newton-Raphson steps accepted: **{nr_count}** ({100*nr_count/(nr_count+bis_count):.1f}%)
- Bisection fallback steps: **{bis_count}** ({100*bis_count/(nr_count+bis_count):.1f}%)

When Newton steps dominate, the hybrid converges as fast as Newton-Raphson.
When Bisection dominates, the function had challenging behavior (flat spots, singularities).
The hybrid never diverges because the bracket is always maintained.
                        """)

        st.divider()
        st.caption(
            "📌 This tool was developed as a Final Year Project — Covenant University, "
            "Department of Mathematics, Industrial Mathematics. Matric: 22CD032166"
        )