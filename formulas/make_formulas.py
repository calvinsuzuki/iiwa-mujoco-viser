"""Render the IK method formulas (LaTeX) to SVGs for the hover previews.

Run:  .venv/bin/python formulas/make_formulas.py
Needs pdflatex (standalone, amsmath, xcolor) and pdftocairo (poppler-utils).
Writes formulas/<key>.svg; iiwa_viser.py embeds them in the page.
Colours are for the dark UI theme (light text on a transparent background).
"""

import pathlib
import subprocess
import tempfile

HERE = pathlib.Path(__file__).resolve().parent

PREAMBLE = r"""
\documentclass[border=6pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage[dvipsnames]{xcolor}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[fontsize=11.5pt]{scrextend}   % text and formulas 15% above the 10pt default
\usepackage[varqu]{zi4}
\usepackage{listings}
\usepackage{enumitem}
\setlist[itemize]{leftmargin=1.1em, itemsep=1pt, topsep=2pt, parsep=0pt, label=\textbullet}
\definecolor{fg}{HTML}{E6EAF0}
\definecolor{accent}{HTML}{FF8A3B}
\definecolor{note}{HTML}{9AA4B0}
\newcommand{\qd}{\dot{q}}
\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
\definecolor{codebg}{HTML}{14171B}
\definecolor{good}{HTML}{3CCF6E}
\definecolor{bad}{HTML}{E5484D}
% Strong / weak points side by side: \proscons{items}{items}, items as \item ...
\newcommand{\proscons}[2]{\par\vspace{6pt}{\small
\begin{minipage}[t]{0.485\linewidth}\raggedright{\color{good}\textbf{Strong}}
\begin{itemize}#1\end{itemize}\end{minipage}\hfill
\begin{minipage}[t]{0.485\linewidth}\raggedright{\color{bad}\textbf{Weak}}
\begin{itemize}#2\end{itemize}\end{minipage}}}
\definecolor{kw}{HTML}{FF9D5C}
\definecolor{cmt}{HTML}{7F8A96}
\definecolor{num}{HTML}{8FD3A8}
\lstset{language=Python, basicstyle=\ttfamily\fontsize{9}{11}\selectfont\color{fg}, breaklines=true,
  keywordstyle=\color{kw}, commentstyle=\color{cmt}\itshape, stringstyle=\color{num},
  backgroundcolor=\color{codebg}, frame=single, rulecolor=\color{codebg},
  framesep=5pt, xleftmargin=5pt, xrightmargin=5pt, columns=fullflexible,
  keepspaces=true, showstringspaces=false, upquote=true,
  morekeywords={True,False,None}}
\begin{document}
\color{fg}
\begin{minipage}{10.5cm}
"""

POST = r"""
\end{minipage}
\end{document}
"""

# Shared footnote: how the commanded twist V comes from the target error.
SERVO = r"""
\vspace{4pt}
{\color{note}\small
$V = \operatorname{sat}(K e)$,\;
$e = \begin{bmatrix} p_d - p \\ \log(R_d R^\top)^\vee \end{bmatrix}$,\;
$K = 5\,\mathrm{s^{-1}}$,\; $q \leftarrow q + \qd\,\Delta t$}
"""

FORMULAS = {
    "pinv": r"""
{\color{accent}\large\textbf{Pseudoinverse}}\par\vspace{4pt}
\[
\qd = J^{+} V,\qquad
J^{+} = J^\top \left(J J^\top\right)^{-1} = \sum_{i=1}^{6} \frac{1}{\sigma_i}\, v_i u_i^\top
\]
Minimum-norm $\qd$ that reproduces $V$ exactly.\par\vspace{2pt}
{\color{note}\small Near a singularity $\sigma_{\min} \to 0$, so $1/\sigma_{\min} \to \infty$:
joint speeds explode. No velocity or joint limits.}
""" + SERVO,

    "dls": r"""
{\color{accent}\large\textbf{Damped least squares}}\par\vspace{4pt}
\[
\Delta q = J^\top \left(J J^\top + \lambda^2 I\right)^{-1} e,
\qquad q \leftarrow q + \Delta q
\]
\[
\Delta q = \arg\min_{\Delta q}\; \norm{J\,\Delta q - e}^2 + \lambda^2 \norm{\Delta q}^2,
\qquad
\frac{1}{\sigma_i} \;\to\; \frac{\sigma_i}{\sigma_i^2 + \lambda^2} \le \frac{1}{2\lambda}
\]
Iterated until $\norm{e} < 10^{-3}$ (max.\ 200), $\lambda = 0.1$.\par\vspace{2pt}
{\color{note}\small Gain stays bounded, but the result is inexact near singularities and
the full pose is re-solved every frame, so $q$ can jump. No limits.}
""",

    "qp-lsq": r"""
{\color{accent}\large\textbf{Constrained least squares}}\par\vspace{4pt}
\[
\min_{\qd}\; \norm{J \qd - V}^2
\quad\text{s.t.}\quad \qd_{lo} \le \qd \le \qd_{hi}
\]
\[
\begin{aligned}
\qd_{lo} &= \max\!\left(-\qd_{\max},\; \tfrac{q_{\min} - q}{\Delta t},\; \qd_{prev} - \ddot q_{\max}\Delta t\right)\\
\qd_{hi} &= \min\!\left(\phantom{-}\qd_{\max},\; \tfrac{q_{\max} - q}{\Delta t},\; \qd_{prev} + \ddot q_{\max}\Delta t\right)
\end{aligned}
\]
{\color{note}\small Velocity, joint-range and acceleration limits always hold. When one is
active $J\qd \ne V$: the closest twist is not in the commanded direction, so the tool drifts off its line.}
""" + SERVO,

    "qp-scaled": r"""
{\color{accent}\large\textbf{Velocity scaling QP}}\par\vspace{4pt}
\[
\max_{\qd,\,\alpha}\; \alpha
\;-\; w \norm{N\left(\qd - \qd_{null}\right)}^2
\;-\; w_s \norm{J_s \qd - V_s}^2
\]
\[
\text{s.t.}\quad J_h \qd = \alpha V_h,\qquad 0 \le \alpha \le 1,\qquad \qd_{lo} \le \qd \le \qd_{hi}
\]
\[
N = I - J^{+} J \quad \text{(null-space projector)}
\]
Tool moves exactly along $V$, only slower ($\alpha < 1$) when a limit or singularity is near.
$\qd_{null}$ swings the elbow to its setpoint without moving the tool.\par\vspace{2pt}
{\color{note}\small $h$: rows held exactly, $s$: soft rows (reach flexibility / restrain only Z-axis).
$w = 0.01$, $w_s = 0.5$. Same limit box as constrained least squares.}
""" + SERVO,
}


# Strong / weak points, shown between the formulas and the code.
PROS_CONS = {
    "pinv": r"""
\proscons{
\item Exact: reproduces $V$ with the smallest $\qd$
\item Closed form, one SVD: cheapest method
\item Easy to analyse ($\sigma_i$ show everything)
}{
\item Explodes at singularities
\item Ignores joint and speed limits
}
""",
    "dls": r"""
\proscons{
\item Bounded gain: no blow-up near singularities
\item Converges to the full pose even from far away
\item Simple, no solver needed
}{
\item Inexact near singularities: $\lambda$ trades accuracy for safety
\item Re-solves the pose each frame: $q$ can jump
\item No limits
}
""",
    "qp-lsq": r"""
\proscons{
\item Speed, range and acceleration limits always hold
\item Smooth, never explodes
\item Any extra linear constraint can be added
}{
\item With a limit active the tool leaves its path
\item Needs a QP solve every step
}
""",
    "qp-scaled": r"""
\proscons{
\item Tool stays on the commanded path, only slower
\item All limits hold, like constrained LS
\item Null space: steers the elbow for free
}{
\item Stops dead ($\alpha \to 0$) at a limit instead of finding a way around
\item Most expensive: 8-variable QP with weights to tune
}
""",
}


CODE_TITLE = r"""
\vspace{8pt}
"""

CODE = {
    "pinv": r"""
\begin{lstlisting}
qd = np.linalg.pinv(J) @ V    # J^+ V, minimum-norm solution
q = q + qd * dt               # integrate, no limits
\end{lstlisting}
""",
    "dls": r"""
\begin{lstlisting}
q = q0.copy()
for _ in range(200):
    pos, quat, J = fk_jac(q)
    e = pose_error(pos, quat, pos_d, quat_d)
    if np.linalg.norm(e) < 1e-3:
        break
    A = J @ J.T + lam**2 * np.eye(6)    # damped
    q = q + J.T @ np.linalg.solve(A, e)
\end{lstlisting}
""",
    "qp-lsq": r"""
\begin{lstlisting}
lo, hi = joint_velocity_box(q, qd_prev, dt, lim)
# ||J qd - V||^2  ->  1/2 qd' P qd + c' qd
P = 2 * J.T @ J
c = -2 * J.T @ V
qd = solve_qp(P, c, A=np.eye(7), l=lo, u=hi)    # OSQP
q = q + qd * dt
\end{lstlisting}
""",
    "qp-scaled": r"""
\begin{lstlisting}
N = np.eye(7) - np.linalg.pinv(J) @ J    # null-space projector
# decision vector x = [qd (7), alpha]
P = np.zeros((8, 8))
P[:7, :7] = 2*w*N + 2*ws * Js.T @ Js
# last entry -1: minimising -alpha = maximising alpha
c = np.r_[-2*w * N @ qd_null - 2*ws * Js.T @ Vs, -1.0]
# rows: Jh qd - alpha Vh = 0, then box on [qd, alpha]
A = np.vstack([np.c_[Jh, -Vh], np.eye(8)])
l = np.r_[np.zeros(len(Vh)), lo, 0.0]
u = np.r_[np.zeros(len(Vh)), hi, 1.0]
qd, alpha = np.split(solve_qp(P, c, A, l, u), [7])
\end{lstlisting}
""",
}


def render(key, body, out_dir=HERE):
    with tempfile.TemporaryDirectory() as tmp:
        tex = pathlib.Path(tmp) / f"{key}.tex"
        tex.write_text(PREAMBLE + body + PROS_CONS[key] + CODE_TITLE + CODE[key] + POST)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                       cwd=tmp, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["pdftocairo", "-svg", f"{key}.pdf", str(out_dir / f"{key}.svg")],
                       cwd=tmp, check=True)


if __name__ == "__main__":
    for key, body in FORMULAS.items():
        render(key, body)
        print("wrote", HERE / f"{key}.svg")
