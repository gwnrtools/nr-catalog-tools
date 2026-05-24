## Comparison of promiment BBH waveform catalogs
### Motivation
There are now many
families of waveform models that groups around the world have been developing
using different underlying formulations of compact binary dynamics. Some of
these include the effective-one-body (EOB) framework, phenomenological 
waveforms, ESIGMA models, self-force - PN hybrid frameworks, and more. Many of
these are calibrated closely to reproduce simulations from one or the other
NR binary black hole merger catalogs. Since various NR catalogs are deeply
relied upon as references, it is important that their underlying errors
themselves be quantified in order to quantify the fundamental limit that our
analytical models will face if calibrated against any one particular NR catalog
versus another.
Multiple suitable catalogs of binary black hole merger simulations are available
in public domain. These include the SXS catalog which uses the Spectral Einstein
Code (SpEC), the RIT binary black hole catalog which uses the LazEv code,
the Georgia Tech / UT Austin catalog which uses the MayaKranc code, and other more
recent ones. 

One clear testbed for measuring the
accuracy of simulations in a catalog would be against simulations from another
code that are supposed to reproduce the exact same binary merger but with
independent formulations of Einstein equations, and different numerical schemes.
These catalogs are, however, inherently difficult to compare against each
other due to the initial conditions being extremely hard to map between different
codes. Due to this we end with a situation where, lets say we take a simulation
$h^a(\theta_i)$ from catalog $A$, with $i \in [0, N(A)-1]$, and find its 
_closest_ counterpart in catalog $B: h^b_{\theta_j}$, with $j\in [0, N(B)-1]$,
 by minimizing $|h^a(\theta_i)-h^b(\theta_j)|$, then the
difference between these two can easily be dominated or at least be comparable
to $|h^a(\theta_i)-h^a(\theta_j)|$ or $|h^b(\theta_i)-h^b(\theta_j)|$.
In other words, catalogs produced by different NR codes are accurate to
the level of the differences produced in waveforms due to the small but
measurable and unavoidable difference in initial conditions that arise due to
inherently different mathematical and code choices made by different NR codes
(for e.g. initial orbital eccentricity is notoriously difficult to match
between two simulations from two different codes).

Additionally, repeatedly performing fine-tuned comparable sets of simulations
will also result in wastage of computing power as different catalogs will need
to become repetitive of each other instead of being entirely complementary in
terms of their parameter space coverage. This puts us in a situation where
continuous testing quickly becomes practically undesirable.
Large-scale efforts happened in the past where directed speclized effort was
made to study simulations from different NR codes on a level field by 
(a) using them as synthetic signals to create simulated GW detector data, which
were them attempted to be found and characterized using cutting-edge algorithms
and waveform models (cite NINJA-1,2), and (b) by studying their intrinsic
errors using identical analysis codes (cite NRAR). While being very informative
exercises, these did not compare NR catalogs with each other directly.
More recently, we have seen the advent of surrogate models of Numerical Relativity
waveforms. These are sophisticated interpolants of waveform multipoles that
can interpolate across a sizable region of the space of initial binary conditions. These
can be remarkable tools enabling cross-code / cross-catalog comparisons, as 
we can now eliminate the need to substitute $\{|h^a(\theta_i)-h^b(\theta_i)|,\,
\forall\, i\in [0,N(A)-1]\}$ with $\{|h^a(\theta_i)-h^b(\theta_j)|,\,\forall\, j\in [0,N(B)-1]\}$,
where we label $B$ as the catalog for which we have an interpolating surrogate
model available. This is especially timely at the moment, since the whole NR 
community has come to model significantly more complex binary configurations
since the NINJA and NRAR efforts concluded.

### Source frame ambiguity
Lets start with realizing that what we have available with us are multipoles
$h_{\ell m}$ of the waveform decomposed at null infinity $\mathcal{I}^+$ in spin
$(-2)$ weighted spherical harmonics in a _source_ frame ${\bf F}_s$, whose
$z$-axis is aligned with some definition of angular momentum of the binary at
the start of the simulation. Let us now say that we have a fixed
detector available for the entire duration of a given signal on Earth. The
polarizations incident on it can be constructed in terms of _source frame_
$h_{\ell m}$ as

$$H(t) = h_+(t) + i h_\times(t) = \Sigma_{\ell, m} {}^{-2}Y^S_{\ell m}(\iota,
\phi_c)\, h^S_{\ell, m}(t; \vec{\theta}), $$

where $\vec{\theta}$ is the set of intrinsic parameters of the binaray, i.e.
$\{m_1, m_2, \vec{\chi}_1, \vec{\chi}_2, \phi_0\}$. This _source frame_ 
${\bf F}_s$ in which the spherical harmonis are constructed is such that the
line of sight in this frame from its origin to the detector has polar angles
$(\iota, \phi_c)$. It is possible for ${\bf F}_s$ to change orientation during
the binary evolution, for e.g. in the presence of spin-orbit coupling driven
orbital-plane precession. In that case, the inclination angles $\iota$ becomes
a function of time. However for NR catalogs, the most common choice is to use
a fixed inertial frame of reference for defining the waveform multipoles. In
such a frame, its the multipoles themselves which encode all the time-dependent
dynamics of the binary and its radiation as seen from an observer located at
$(\iota, \phi_c)$ in this fixed frame.
Different numerical relativity groups adopt distinct conventions for defining this source frame $\mathbf{F}_s$. For instance, one code might align the $z$-axis with the instantaneous orbital angular momentum at the relaxation time, while another might align it with the initial Newtonian angular momentum or the principal axes of the computational grid. Consequently, even for identical intrinsic parameters $\vec{\theta}$, the decomposed multipoles $h_{\ell m}$ will differ solely due to these coordinate choices. This introduces an artificial frame mismatch that must be disentangled from true physical or numerical discrepancies.To perform a robust comparison using the surrogate model, we must treat the frame orientation and time/phase definitions as nuisance parameters to be marginalized over. Since the surrogate model allows us to generate the waveform $h^b(\vec{\theta})$ at the exact same intrinsic parameters as the catalog simulation $h^a(\vec{\theta})$, we effectively eliminate the parameter space distance error described earlier. The remaining task is to find the optimal frame transformation that aligns the surrogate prediction with the catalog simulation.
To compare a waveform from a generic precessing binary simulation (Catalog Source Frame $\mathbf{F}_C$) with a Surrogate model (Surrogate Source Frame $\mathbf{F}_S$), we must mathematically construct a transformation that maps the surrogate's multipoles into the catalog's frame of reference. Here we assume that both $\mathbf{F}_C$ and $\mathbf{F}_S$ are inertial frames which are fixed in time amd are related by a static rigid rotation. 
This transformation is a composition of a spatial rotation, a time translation, and a phase shift. Let the gravitational-wave strain in the original Surrogate frame, at a time $t$ and sky direction $\mathbf{n}_S = (\theta_S, \phi_S)$, be defined by the multipole expansion:$$h^S(t, \mathbf{n}_S) = \sum_{\ell=2}^{\infty} \sum_{m=-\ell}^{\ell} h^S_{\ell m}(t)\, {}^{-2}Y_{\ell m}(\mathbf{n}_S)$$where ${}^{-2}Y_{\ell m}$ are the spin-weighted spherical harmonics of weight $-2$.
We posit that the Catalog frame $\mathbf{F}_C$ differs from the Surrogate frame $\mathbf{F}_S$ by a generic rigid rotation $R \in SO(3)$. This rotation accounts for differences in how various NR codes define the "z-axis" (e.g., along the initial orbital angular momentum $\mathbf{L}_0$ versus the remnant spin $\mathbf{\chi}_f$) and the "x-axis" (e.g., along the initial separation vector or the line of nodes).
If $R$ is the active rotation operator that rotates the physical system (or equivalently, the passive rotation of the coordinate axes from $\mathbf{F}_S$ to $\mathbf{F}_C$), a point on the sky $\mathbf{n}$ has coordinates related by $\mathbf{n}_S = R^{-1} \mathbf{n}_C$. The scalar field property of the strain implies $h(\mathbf{n}_C) = h(R\,\mathbf{n}_S)$. Under such a rotation, the spin-weighted spherical harmonics transform linearly into each other within the same $\ell$-subspace. The standard transformation happens under the $(2\ell+1)$-dimensional irreducible representation of the rotation group $SO(3)$ law for the expansion coefficients $h_{\ell m}$, governed by the Wigner D-matrices $D^{\ell}_{m' m}(R)$:

$$h^{S, \text{rot}}_{\ell m}(t) = \sum_{m'=-\ell}^{\ell} h^S_{\ell m'}(t)\, D^{\ell}_{m' m}(R)$$

**Note on Precession:** For non-precessing binaries, the waveform is dominated by the $(\ell, \pm 2)$ modes, and the mixing is often trivial. However, for a precessing binary, the orbital plane wobbles, distributing power across all $m$-modes. Consequently, the summation over $m'$ above is physically significant; it captures the "mode-mixing" required to project the wobbling signal onto the new, fixed detector frame.
The catalog simulation may define its merger time $t=0$ differently from the surrogate (e.g., peak amplitude vs. peak frequency). We introduce a time shift parameter $t_c$ such that the surrogate's time coordinate $t$ maps to $t - t_c$:$$h^{S, \text{shifted}}_{\ell m}(t) = h^{S, \text{rot}}_{\ell m}(t - t_c)$$

Finally, we allow for an arbitrary orbital phase offset $\phi_c$. In the context of the decomposed multipoles, a phase rotation of the binary by $\phi_c$ around the reference $z$-axis corresponds to a complex rotation of the modes:$$h^S_R \to e^{-i m \phi_c} h^{S, \text{shifted}}_{\ell m}$$

Combining these steps, we arrive at the full expression for the surrogate waveform $h^S_R$ in the catalog's frame. This is the "target" model we utilize in our mismatch minimization:
$$\boxed{
h^S_{R, \ell m}(t; t_c, \phi_c, R) = e^{-i m \phi_c} \sum_{m'=-\ell}^{\ell} h^S_{\ell m'}(t - t_c)\, D^{\ell}_{m' m}(R)
}$$

In this equation:
- $t_c$ aligns the temporal peaks of the signal.
- $R$ (parameterized by quaternions or Euler angles $\alpha, \beta$) aligns the orientation of the binary's average orbital plane (the "tilt").
- $\phi_c$ aligns the orbital phase (the "twist").
- The sum over $m'$ represents the crucial mode-mixing operation that enables accurate comparison of precessing waveforms where no single frame is "inertial" with respect to the binary's radiation pattern.
### BMS Symmetries and Supertranslations


While the frame rotations discussed previously address the $SO(3)$ rotational ambiguity of the source, a more subtle class of ambiguities arises from the structure of null infinity itself. The asymptotic symmetry group of General Relativity at null infinity $\mathcal{I}^+$ is the BMS group (Bondi-Metzner-Sachs). The BMS group is a semi-direct product of the Lorentz group and an infinite-dimensional abelian group known as supertranslations.Physically, a supertranslation corresponds to a direction-dependent shift in the retarded time coordinate $u$. Unlike a standard time translation where $u \to u + \delta t$ (a constant shift for all angles), a supertranslation shifts time differently depending on the observation angle $(\theta, \phi)$:$$u' = u - \alpha(\theta, \phi)$$where $\alpha(\theta, \phi)$ is an arbitrary smooth function on the sphere.The standard Poincaré group is a subgroup of BMS.
- $\ell = 0$ mode of $\alpha$: Corresponds to a standard time translation (discussed previously as $t_c$).
- $\ell = 1$ modes of $\alpha$: Correspond to spatial translations (changing the origin of the coordinate system).
- $\ell \ge 2$ modes of $\alpha$: correspond to "proper" supertranslations.

Different NR codes may implicitly slice null infinity with different "supertranslation frames" due to their extrapolation methods or Cauchy-characteristic extraction techniques. Consequently, two waveforms $h^A$ and $h^B$ describing the exact same physical binary may differ because $h^B$ is effectively supertranslated relative to $h^A$. To compare them accurately, we must map the surrogate waveform not just by a rotation $R$, but also by a supertranslation field $\alpha(\theta, \phi)$.

**Derivation of the Supertranslated Waveform Modes** We seek the transformation of the waveform modes $h_{\ell m}(u)$ under the mapping $u \to u - \alpha(\theta, \phi)$.Let the original strain be $h(u, \theta, \phi)$. The transformed strain $h'(u, \theta, \phi)$ is:$$h'(u, \theta, \phi) = h(u - \alpha(\theta, \phi), \theta, \phi)$$For small supertranslations (which is physically appropriate for comparing NR catalogs that differ only by numerical gauge choices), we can perform a Taylor expansion around the original time $u$:$$h'(u, \theta, \phi) \approx h(u, \theta, \phi) - \alpha(\theta, \phi) \, \dot{h}(u, \theta, \phi)$$where $\dot{h} = \partial h / \partial u$.We now decompose both the supertranslation field $\alpha$ and the waveform $h$ into their respective spherical harmonics.The supertranslation field is a scalar, so we expand it in standard scalar spherical harmonics $Y_{\ell m}$:$$\alpha(\theta, \phi) = \sum_{j, k} \alpha_{j k} \, Y_{j k}(\theta, \phi)$$The strain is a spin-weight $-2$ field, expanded in spin-weighted harmonics:$$h(u, \theta, \phi) = \sum_{\ell, m} h_{\ell m}(u) \, {}^{-2}Y_{\ell m}(\theta, \phi)$$Substituting these into the Taylor expansion:$$\sum_{\ell, m} h'_{\ell m}(u) \, {}^{-2}Y_{\ell m} = \sum_{\ell, m} h_{\ell m}(u) \, {}^{-2}Y_{\ell m} - \left( \sum_{j, k} \alpha_{j k} Y_{j k} \right) \left( \sum_{p, q} \dot{h}_{p q}(u) \, {}^{-2}Y_{p q} \right)$$To isolate the transformed mode coefficients $h'_{\ell m}$, we rely on the orthonormality of the spin-weighted spherical harmonics. We multiply both sides by ${}^{-2}Y_{\ell m}^*$ and integrate over the sphere $d\Omega$:$$h'_{\ell m}(u) = h_{\ell m}(u) - \sum_{j, k} \sum_{p, q} \alpha_{j k} \, \dot{h}_{p q}(u) \int_{S^2} {}^{-2}Y_{\ell m}^*(\Omega) \, Y_{j k}(\Omega) \, {}^{-2}Y_{p q}(\Omega) \, d\Omega$$The integral on the right is the Gaunt integral (or a product of three spin-weighted spherical harmonics), which can be evaluated analytically in terms of Clebsch-Gordan coefficients. Let us denote this geometric coupling term as $\mathcal{G}^{\ell m}_{j k, p q}$.The final mode-mixing transformation due to a supertranslation is:$$\boxed{ h'_{\ell m}(u) = h_{\ell m}(u) - \sum_{j=0}^{j_{max}} \sum_{k=-j}^{j} \sum_{p, q} \alpha_{j k} \, \mathcal{G}^{\ell m}_{j k, p q} \, \dot{h}_{p q}(u) }$$


#### Implications for Catalog Comparison

This result demonstrates that supertranslations induce mode mixing proportional to the time derivative of the waveform.
- $\ell=1$ (Center of Mass Shift): The most dominant error in NR comparisons is often a displacement of the coordinate origin. This corresponds to optimizing the coefficients $\alpha_{1m}$. This effectively "shifts" the waveform modes to correct for the center-of-mass drift or offset.
- Higher Order ($\ell \ge 2$): While typically smaller, differences in how NR codes handle the "corners" of the computational grid or the extraction surface can introduce higher-order supertranslation biases.

To achieve the highest precision mismatch $\mathcal{M}$, we should extend the optimization vector $x$ to include not just the rotation angles $(\alpha, \beta)$ and time shift $t_c$ (which is $\alpha_{00}$), but also the spatial translation coefficients $\alpha_{1m}$.
### Quantifying waveform disagreements


We now define the agreement between the two waveforms via the standard noise-weighted inner product $\langle \cdot | \cdot \rangle$. For two waveforms $h_1$ and $h_2$, the overlap is given by:

$$\mathcal{O}(h_1, h_2) = \frac{\langle h_1 | h_2 \rangle}{\sqrt{\langle h_1 | h_1 \rangle \langle h_2 | h_2 \rangle}}$$

where the inner product is defined as $\langle h_1 | h_2 \rangle = 4 \Re \int_{f_{min}}^{f_{max}} \frac{\tilde{h}_1(f) \tilde{h}^*_2(f)}{S_n(f)} df$. Here, $\tilde{h}(f)$ denotes the Fourier transform of the time-domain strain, and $S_n(f)$ is the power spectral density of the detector noise (or a flat spectrum if a purely numerical comparison is desired).The comparison metric of interest is the mismatch (or unfaithfulness), $\mathcal{M}$, which quantifies the disagreement between the catalog simulation and the surrogate model. To compute this, we must maximize the overlap over the extrinsic parameters: time of arrival $t_c$, coalescing phase $\phi_c$, and the spatial rotation parameters defining the frame $\mathbf{F}(\rho, \sigma)$ (often parameterized via Euler angles or a quaternion rotation $R$). The mismatch is thus defined as:

$$\mathcal{M} = 1 - \max_{t_c, \phi_c, R \in SO(3)} \left[ \frac{\langle h^a(\vec{\theta}) | h^S_R(\vec{\theta}, t_c, \phi_c) \rangle}{\sqrt{\langle h^a | h^a \rangle \langle h^S | h^S \rangle}} \right]$$

where $h^S_R$ denotes the surrogate waveform transformed by the rotation $R$ and shifted in time and phase. By minimizing this mismatch, we isolate the intrinsic numerical error of the catalog simulation relative to the consensus represented by the surrogate, decoupled from the arbitrary choices of initial frames and coordinate times. This methodology transforms the catalog comparison problem from a discrete nearest-neighbor search into a continuous optimization problem in the extrinsic parameter space, providing a far more stringent test of waveform accuracy.
