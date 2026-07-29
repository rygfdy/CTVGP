# Additional rebuttal proofs

This note collects the detailed arguments referenced in the final responses
but not repeated in the main complexity-and-fidelity document.

## 1. Centered finite differences converge to GP derivatives

Let \(f\) be a mean-square differentiable Gaussian process and let

\[
\Delta_\delta f(z)
=
\frac{f(z+\delta)-f(z-\delta)}{2\delta}.
\]

Mean-square differentiability at \(z\) means

\[
\lim_{t\to0}
\mathbb E\!\left[
\left\{
\frac{f(z+t)-f(z)}{t}-f'(z)
\right\}^{2}
\right]=0.
\]

Write the centered difference as the average of a forward and a backward
difference:

\[
\Delta_\delta f(z)-f'(z)
=
\frac12
\left\{
\frac{f(z+\delta)-f(z)}{\delta}-f'(z)
\right\}
+
\frac12
\left\{
\frac{f(z)-f(z-\delta)}{\delta}-f'(z)
\right\}.
\]

Using \((a+b)^2\le 2a^2+2b^2\), the expected squared discrepancy is bounded by
one half of the sum of the forward and backward mean-square derivative
errors. Both terms converge to zero, hence

\[
\mathbb E[
\{\Delta_\delta f(z)-f'(z)\}^{2}
]\longrightarrow0.
\]

For the RBF kernel, all kernel derivatives exist and the process is
mean-square differentiable to every order, so the condition holds. The GP
derivative and the finite difference are correlated Gaussian linear
functionals; equality is asymptotic in mean square, not deterministic at a
fixed nonzero \(\delta\).

## 2. Derivative positivity and discrete isotonicity

If \(f\) is absolutely continuous on every grid interval and
\(f'(x)\ge0\) almost everywhere, then for \(x_i<x_{i+1}\),

\[
f(x_{i+1})-f(x_i)
=
\int_{x_i}^{x_{i+1}} f'(t)\,dt
\ge0.
\]

Thus global derivative positivity implies isotonicity on every ordered grid.
The converse is false: discrete endpoint inequalities constrain only interval
integrals and allow a negative derivative on a subinterval.

Finite derivative constraints also do not imply global monotonicity without a
regularity margin. Suppose \(f'\) is \(L\)-Lipschitz and the derivative
constraint locations \(Z=\{z_j\}\) have fill distance

\[
\delta_Z=\sup_x\min_j|x-z_j|.
\]

For any \(x\), choose \(z_j\) with \(|x-z_j|\le\delta_Z\). Then

\[
f'(x)
\ge f'(z_j)-L|x-z_j|
\ge \min_j f'(z_j)-L\delta_Z.
\]

Consequently, the sufficient margin condition

\[
\min_j f'(z_j)\ge L\delta_Z
\]

implies global derivative nonnegativity. Without either a global constraint
or such a margin/regularity condition, finite-node derivative positivity is
only a finite-grid guarantee.

## 3. Support guarantee with an inducing Gaussian block

Let an inducing approximation produce any Gaussian derivative block

\[
\widetilde D\sim
N(\widetilde m_d,\widetilde\Sigma_{dd}).
\]

CTVGP fits an auxiliary Gaussian \(\Xi\) to the hard target defined by this
approximate block and generates

\[
D=h(\Xi),
\]

where every coordinate map satisfies \(h_j:\mathbb R\to(0,\infty)\). Therefore

\[
Q_\theta(D\in\mathbb R_+^{m_d})=1
\]

algebraically, regardless of the negative mass of the pre-transport Gaussian
block. The inducing approximation can change fidelity to the dense-GP hard
posterior because its mean and covariance are approximate, but it cannot
create a post-transport support violation.

## 4. Exponential maps and the stochastic-gradient condition

For \(h(u)=e^u\), the bounded-slope assumption is unavailable. Derivatives of
the reparameterized Gaussian target and Jacobian terms can instead be bounded
by expressions of the form

\[
C(1+\|\epsilon\|^k)
\exp\{c\|\xi_\theta(\epsilon)\|_\infty\}
\]

for finite \(C,c,k\). A sufficient uniform-integrability replacement is

\[
\sup_{\theta\in\Theta}
\mathbb E\!\left[
(1+\|\epsilon\|^k)
\exp\{c\|\xi_\theta(\epsilon)\|_\infty\}
\right]<\infty
\]

for every finite \(c,k\) required by the first- and second-order derivative
bounds.

If the auxiliary means and covariance parameters range over a compact set and
the covariance eigenvalues are uniformly bounded, then every coordinate of
\(\xi_\theta(\epsilon)\) is Gaussian with uniformly bounded mean and variance.
Gaussian moment-generating functions are finite for every finite linear
exponent. Polynomial factors are absorbed by slightly enlarging the
exponential moment, so the displayed expectation is uniformly finite.

Together with unbiased stochastic gradients, bounded conditional second
moments, a Lipschitz population gradient, and Robbins--Monro step sizes, the
standard stochastic-approximation argument then yields

\[
\liminf_{t\to\infty}
\|\nabla\mathcal L(\theta_t)\|=0
\quad\text{almost surely}.
\]

This extends the stationarity argument under stronger moment and
parameter-compactness assumptions; it is not a bounded-slope theorem.

## 5. Jacobian of a coordinatewise positive map

For \(h(\xi)=(h_1(\xi_1),\ldots,h_{m_d}(\xi_{m_d}))\),

\[
J_h(\xi)
=
\operatorname{diag}
\{h_1'(\xi_1),\ldots,h_{m_d}'(\xi_{m_d})\}.
\]

If \(h_j'(\xi_j)>0\) for every coordinate, then

\[
\det J_h(\xi)
=
\prod_{j=1}^{m_d} h_j'(\xi_j)>0.
\]

Hence the inverse-map density and log-Jacobian used by CTVGP are well defined
throughout the positive orthant.

## 6. Scope of multivariate convexity

Full multivariate convexity constrains Hessian matrices to the
positive-semidefinite cone, not coordinatewise entries to an orthant. A
cone-adapted construction could use

\[
H_j=L_jL_j^\top
\]

with triangular \(L_j\) and a positive diagonal. However, this requires a
different Jacobian, parameterization, optimization study, and empirical
coverage analysis. The rebuttal therefore treats full PSD-cone convexity as
future work rather than as an implemented contribution.

