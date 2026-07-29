# Complexity and Fidelity of Conditional-Transport Variational GPs

## Mini-outline

- Localize the joint approximation error to the constrained derivative marginal.
- Separate support correctness from finite-distance approximation quality.
- State the exact rank-nesting result and explain why dimension alone cannot imply degradation.
- Give a conditional covariance-capacity lower bound with explicit spectral assumptions.
- Analyze how the positive map changes auxiliary-tail geometry.
- Derive implementation-level complexity in \(n,m_d,r,B,S\).
- Translate fitted KL into TV, event-probability, Wasserstein, and predictive bounds.

## 1. Setup

Let \(D\in\mathbb R^{m_d}\) collect the constrained GP derivatives and let
\[
P_h(d)=\frac{\phi_{m_d}(d;m_d,\Sigma_{dd})\mathbf 1\{d\ge 0\}}{Z_h},
\qquad
Z_h=\Pr_{N(m_d,\Sigma_{dd})}(D\ge 0).
\]
CTVGP draws an auxiliary Gaussian \(\Xi\sim N(\mu_\xi,\Sigma_\xi)\), with either
\[
\Sigma_\xi=\operatorname{diag}(s^2)+UU^\top,\quad U\in\mathbb R^{m_d\times r},
\]
or an unrestricted full covariance, and applies a coordinatewise positive map
\(D=h(\Xi)\). The resulting derivative density is
\[
q_\theta(d)
=
\phi_{m_d}\!\left(h^{-1}(d);\mu_\xi,\Sigma_\xi\right)
\prod_{j=1}^{m_d}
\left|\frac{d}{dd_j}h^{-1}(d_j)\right|,
\qquad d>0.
\]
The notation \(m_d\) denotes the derivative-vector dimension. In the present
one-dimensional experiments it equals the number of constraint locations; in
multiple input dimensions it equals “number of locations \(\times\) number of
constrained directions per location.”

For convergence diagnostics, \(\theta\) denotes parameters standardized by the
posterior derivative scale (means and low-rank rows use their delta-method
auxiliary scale; log-scales are already dimensionless). Thus
\(\|\nabla_\theta\mathcal L\|/(1+\|\theta\|)\) is invariant to a change of
derivative measurement units. This coordinate convention is fixed for all
cases and restarts.

## 2. Exact localization of the joint KL

**Proposition 1 (exact KL localization).** Suppose the target and approximation
share the exact GP conditional,
\[
P(f,d)=P(f\mid d)P_h(d),
\qquad
Q_\theta(f,d)=P(f\mid d)Q_\theta(d).
\]
Whenever \(Q_\theta\ll P_h\),
\[
\mathrm{KL}\!\left(Q_\theta(f,d)\,\|\,P(f,d)\right)
=
\mathrm{KL}\!\left(Q_\theta(d)\,\|\,P_h(d)\right).
\]

**Proof.** By the chain rule for relative entropy,
\[
\begin{aligned}
\mathrm{KL}(Q_\theta(f,d)\|P(f,d))
&=\int Q_\theta(d)P(f\mid d)
\log\frac{Q_\theta(d)P(f\mid d)}{P_h(d)P(f\mid d)}\,df\,dd\\
&=\int Q_\theta(d)\log\frac{Q_\theta(d)}{P_h(d)}\,dd,
\end{aligned}
\]
because \(\int P(f\mid d)\,df=1\). \(\square\)

This identity makes a direct \(q_\theta\)-versus-\(P_h\) experiment the correct
fidelity diagnostic for the full conditional GP construction; it is not merely
a proxy based on predictive moments.

## 3. Support mismatch and what the distance table means

**Proposition 2 (Gaussian support mismatch).** Let \(G\) be any non-degenerate
Gaussian distribution on \(\mathbb R^{m_d}\). If
\(\Pr_G(D\not\ge0)>0\), then
\[
\mathrm{KL}(G\|P_h)=+\infty.
\]

**Proof.** The hard target assigns probability zero to
\(\mathbb R^{m_d}\setminus\mathbb R_+^{m_d}\), whereas \(G\) assigns this set
positive probability. Hence \(G\not\ll P_h\), and relative entropy is infinite
by definition. \(\square\)

Consequently, Standard GP, Projection GP (whose derivative marginal is
unchanged), and non-degenerate Gaussian-site EP have infinite reverse KL. This
does not make their forward KL or TV undefined. In particular, if \(G\) is the
unconstrained GP derivative marginal, then
\[
\operatorname{TV}(G,P_h)=1-Z_h.
\]
The deterministic Basis GP derivative is a point mass. It and the continuous
\(P_h\) are mutually singular, so both KL directions are infinite and TV is
one, although Wasserstein distances remain finite.

## 4. Rank nesting and the limit of dimension-only claims

Define \(\mathcal Q_r\) as the transported family whose auxiliary covariance is
diagonal plus rank \(r\). Padding \(U\) with one zero column leaves the
distribution unchanged, hence
\[
\mathcal Q_r\subseteq\mathcal Q_{r+1}.
\]

**Proposition 3 (population rank monotonicity).** For every fixed hard target,
\[
\inf_{Q\in\mathcal Q_{r+1}}\mathrm{KL}(Q\|P_h)
\le
\inf_{Q\in\mathcal Q_r}\mathrm{KL}(Q\|P_h).
\]

This is a statement about the population optimum, not a guarantee about a
finite-run optimizer. If the best observed validation ELBO decreases after
enlarging the family by more than combined Monte Carlo uncertainty, the correct
interpretation is optimization failure, not structural rank degradation.

**Why fixed \(r\) need not worsen solely because \(m_d\) increases.** There is
no target-independent monotonic theorem in \(m_d\). Adding a constraint may be
nearly redundant (for example, a derivative at a location arbitrarily close to
an existing location), or it may be essentially inactive because its Gaussian
mean is many marginal standard deviations above zero. The incremental
information and incremental approximation error can then be arbitrarily small.
More directly, two target sequences of different dimensions can have entirely
different means and covariances: a one-dimensional target concentrated at the
boundary can be harder for a fixed positive map than a higher-dimensional,
almost-untruncated independent target. Thus \(m_d\) alone does not order the
optimal KL. Any growth claim requires an explicit growth condition on target
dependence or covariance residual.

## 5. A conditional covariance-capacity lower bound

The following result isolates the role of rank without claiming that the
positive-map hard target is Gaussian.

Let an auxiliary Gaussian surrogate be
\(\Pi_m=N(\mu_\star,\Sigma_\star)\), and let
\(\mathcal S_{m,r}\) be the diagonal-plus-rank-\(r\) covariance class, restricted
to matrices with eigenvalues in \([\lambda,\Lambda]\), where
\(0<\lambda\le\Lambda<\infty\). Assume also
\(\lambda I\preceq\Sigma_\star\preceq\Lambda I\). Define
\[
\delta_{m,r}
=
\inf_{\Sigma\in\mathcal S_{m,r}}
\|\Sigma-\Sigma_\star\|_F.
\]

**Theorem 4 (conditional Gaussian-surrogate capacity bound).**
\[
\inf_{\mu,\Sigma\in\mathcal S_{m,r}}
\mathrm{KL}\!\left(
N(\mu,\Sigma)\,\|\,N(\mu_\star,\Sigma_\star)
\right)
\ge
\frac{\lambda^2}{4\Lambda^4}\,\delta_{m,r}^2.
\]

**Proof.** The mean contribution to Gaussian KL is nonnegative, so its optimum
is attained at \(\mu=\mu_\star\). Set
\(A=\Sigma_\star^{-1/2}\Sigma\Sigma_\star^{-1/2}\). Then
\[
\mathrm{KL}
=\frac12\{\operatorname{tr}(A)-m-\log\det A\}.
\]
Every eigenvalue of \(A\) lies in
\([\lambda/\Lambda,\Lambda/\lambda]\). For
\(g(t)=t-1-\log t\), \(g(1)=g'(1)=0\) and
\[
g''(t)=t^{-2}\ge(\lambda/\Lambda)^2
\]
on this interval. Strong convexity therefore gives
\[
\mathrm{KL}\ge
\frac{\lambda^2}{4\Lambda^2}\|A-I\|_F^2.
\]
Finally,
\[
\|A-I\|_F
=
\|\Sigma_\star^{-1/2}(\Sigma-\Sigma_\star)
\Sigma_\star^{-1/2}\|_F
\ge \Lambda^{-1}\|\Sigma-\Sigma_\star\|_F.
\]
Taking the infimum over \(\mathcal S_{m,r}\) proves the claim. \(\square\)

**Corollary 4.1.** If, for a stated target sequence,
\(\delta_{m,r}^2=\Omega(m_d-r)\), then the optimal Gaussian-surrogate KL is
\(\Omega(m_d-r)\).

**Scope.** The theorem is conditional on a Gaussian auxiliary surrogate,
uniform spectral bounds, and an explicitly growing covariance residual. It is
not a universal theorem for all transformed truncated-Gaussian targets. The
capacity experiment tests whether the paper cases empirically display the
postulated residual-growth behavior.

## 6. How the positive map \(h\) changes approximation difficulty

Let \(\pi_h(\xi)=P_h(h(\xi))|h'(\xi)|\) denote the exact auxiliary density in one
coordinate, and suppose \(P_h(d)\to c\in(0,\infty)\) as \(d\downarrow0\).

For softplus \(h(x)=\log(1+e^x)\), as \(x\to-\infty\),
\[
h(x)\sim e^x,\qquad h'(x)\sim e^x,\qquad
\pi_h(x)\sim c e^x.
\]
The exponential map \(h(x)=e^x\) has the same exponential auxiliary left-tail
order. For squareplus
\[
h_\alpha(x)=\frac12\left(x+\sqrt{x^2+\alpha^2}\right),
\]
rationalization gives, as \(x\to-\infty\),
\[
h_\alpha(x)\sim\frac{\alpha^2}{4|x|},
\quad
h_\alpha'(x)\sim\frac{\alpha^2}{4x^2},
\quad
\pi_{h_\alpha}(x)\sim\frac{c\alpha^2}{4x^2}.
\]
Thus squareplus induces a polynomial rather than exponential exact auxiliary
left tail. A Gaussian auxiliary family matches none of these tails exactly, so
there is no map-independent ordering. The maps also create different Jacobian
gradients and place boundary mass at different auxiliary magnitudes. The
pre-registered map experiment therefore reports both distances and Jacobian
tail/gradient diagnostics, with \(\alpha/s_d=0.25\) for squareplus.

## 7. Complexity

Let \(n\) be the number of observations, \(m=m_d\) the derivative dimension,
\(r\) the auxiliary rank, \(B\) the VI batch size, and \(S\) the number of
generated samples.

### 7.1 Parameter count

\[
\begin{array}{ll}
\text{diagonal:} & O(m),\\
\text{diagonal plus rank }r: & O(mr),\\
\text{full covariance:} & O(m^2).
\end{array}
\]

### 7.2 Exact-GP preprocessing

Factoring the \(n\times n\) training covariance, solving for the
training--derivative cross block, forming the derivative Schur complement, and
factoring it cost
\[
O(n^3+n^2m+nm^2+m^3).
\]
This one-time exact-GP cost is separate from transport optimization.

### 7.3 One low-rank VI iteration

For \(\Sigma_\xi=D^2+UU^\top\), v2 uses
\[
\log|\Sigma_\xi|
=2\sum_j\log D_{jj}
+\log|I_r+U^\top D^{-2}U|
\]
and the Woodbury identity. Entropy, its inverse-diagonal term, and
\(\Sigma_\xi^{-1}U\) therefore require
\[
O(mr^2+r^3)
\]
rather than a dense \(O(m^3)\) inverse. Drawing the reparameterized batch and
forming low-rank gradients costs \(O(Bmr)\). However, evaluating the dense
Gaussian target score for \(B\) derivative vectors still costs \(O(Bm^2)\).
Hence the implemented per-iteration total is
\[
\boxed{O(Bm^2+Bmr+mr^2+r^3)}.
\]
Low rank reduces auxiliary covariance work and parameterization, but it cannot
remove the dense-target \(Bm^2\) term.

The previous v1 low-rank implementation explicitly formed
\(D^2+UU^\top\) and computed a dense inverse at every objective evaluation, so
that path contained \(O(m^3)\) work. v2 removes it; the mandatory equivalence
test checks the Woodbury and dense objective/gradient to numerical precision.

### 7.4 Sampling and memory

\[
\begin{array}{ll}
\text{low-rank sampling:} & O(Smr)\quad
(\text{or }O(Sm)\text{ for }r=0),\\
\text{full-covariance sampling:} & O(Sm^2).
\end{array}
\]
Ignoring output storage, low-rank working memory is
\(O(m^2+Bm+mr+Br)\), where the \(m^2\) term is the dense target factor;
the full family additionally stores \(O(m^2)\) variational parameters.

## 8. Turning fitted KL into predictive discrepancy

If \(\varepsilon=\mathrm{KL}(Q_d\|P_h)\), Pinsker's inequality gives
\[
\operatorname{TV}(Q_d,P_h)\le\sqrt{\varepsilon/2}.
\]
Therefore, for every derivative event \(A\),
\[
|Q_d(A)-P_h(A)|\le\sqrt{\varepsilon/2}.
\]

If \(P_h\) is \(\kappa\)-strongly log-concave on its convex support, the
Talagrand \(T_2\) inequality yields
\[
W_2^2(Q_d,P_h)\le \frac{2}{\kappa}\varepsilon,
\qquad
W_1(Q_d,P_h)\le W_2(Q_d,P_h).
\]
For the GP conditional mean
\[
\mu_{f\mid d}=\mu_f+A(d-\mu_d),
\qquad
A=\Sigma_{fd}\Sigma_{dd}^{-1},
\]
any coupling of \(Q_d,P_h\) gives
\[
\|\mathbb E_Q f-\mathbb E_P f\|_2
\le \|A\|_2 W_2(Q_d,P_h)
\le \|A\|_2\sqrt{2\varepsilon/\kappa}.
\]
Analogous Lipschitz predictive functionals inherit the same Wasserstein bound.
These consequences explain why direct derivative fidelity controls the full
conditional construction, while keeping the curvature assumption explicit.

## 9. Reviewer-facing limitations

1. Infinite reverse KL for a Gaussian baseline is a support theorem, not a
   claim that every other metric is poor; forward KL, TV, and Wasserstein must
   also be reported.
2. Rank nesting concerns global population optima. Finite optimization can
   violate the observed ordering, which is why independent restarts and
   combined-MC-SE checks are reported.
3. The \(\Omega(m_d-r)\) statement requires covariance-residual growth and
   bounded spectra. It is not asserted for arbitrary truncated targets.
4. Map-tail analysis predicts different approximation geometry, but it does
   not prove a universal ranking of softplus, exponential, and squareplus.
5. In these experiments \(m_d\) and the number of constraint points coincide;
   they need not coincide in a multidirectional constraint problem.

## 10. Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| Joint KL equals derivative-marginal KL | Proposition 1 | Supported exactly |
| Gaussian baselines have infinite reverse KL | Proposition 2 and support-aware test | Supported exactly |
| Larger rank cannot worsen the population optimum | Proposition 3 | Supported exactly |
| Fixed rank must worsen for every increasing \(m_d\) | Dimension-only counterexamples | Rejected; claim removed |
| KL grows as \(\Omega(m_d-r)\) under residual growth | Theorem 4 and Corollary 4.1 | Supported conditionally |
| v2 removes the low-rank \(m_d^3\) entropy inverse | Woodbury derivation and dense-equivalence test | Supported |
| Low rank removes all quadratic cost | Dense target-score term | Rejected; claim removed |
| Fitted KL controls TV/events and, conditionally, Wasserstein | Pinsker and \(T_2\) consequences | Supported with stated assumptions |

## 11. Adversarial self-review

- **Contribution:** Pass. The section adds a direct error localization result,
  a support-aware interpretation, and an implementation-level rank analysis.
- **Writing clarity:** Pass. \(m_d,r,B,S\), target support, and all conditions
  are defined before use.
- **Experimental strength:** Pending until the full capacity, map, and timing
  CSVs are inserted into the results document.
- **Evaluation completeness:** Pass by design: main baselines, rank families,
  maps, seeds, optimization failures, and timing components are all registered.
- **Method soundness:** Pass with limitations. The dense target remains
  quadratic, and neither dimension nor map admits an unconditional fidelity
  ordering.
