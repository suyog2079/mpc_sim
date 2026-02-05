State Space Formulation
 State:
 $$X(k) = \begin{bmatrix}
x \\
y \\
\theta \\
v_{x} \\
v_{y} \\
\omega \\
a_{x} \\
a_{y}
\end{bmatrix}$$

Input:
$$U(k) = \begin{bmatrix}
v_{x} \\
v_{y} \\
\omega
\end{bmatrix}$$

state update:
$X(k+1) = AX(k) + BU(k)$

$$A = \begin{bmatrix}
1 & 0 & 0 & dt & 0 & 0 & \frac{1}{2}dt^2 &  0 \\
0 & 1 & 0 & 0 & dt & 0 & 0 & \frac{1}{2}dt^2 \\
0 & 0 & 1 & 0 & 0 & dt & 0 & 0 &  \\
0 & 0 & 0 & 0 & 0 & 0 & dt & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & dt \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 &  \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
\end{bmatrix}$$

$$B = \begin{bmatrix}
 0 & 0 & 0 \\
 0 & 0 & 0 \\
 0 & 0 & 0 \\
 1 & 0 & 0 \\
 0 & 1 & 0 \\
 0 & 0 & 1 \\
 0 & 0 & 0 \\
 0 & 0 & 0 
\end{bmatrix}$$


Observation model:
$Y(k) = CX(k)$

$$C = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 
\end{bmatrix} = I_{8}$$


Prediction model:
$Y_{p}(k+1)=S^XX(k) + S^U_{pre} U(k-1)+S^U \Delta U_m(k)$

$$S^X = \begin{bmatrix}
CA \\
CA^2 \\
CA^3 \\
\vdots \\
CA^p
\end{bmatrix}$$
$$S^U_{pre} = \begin{bmatrix}
CB \\
CB+CAB \\
CB + CAB + CA^2B \\
\vdots \\
\displaystyle\sum_{i=0}^{p-1} CA^iB
\end{bmatrix}$$

$$S^U = \begin{bmatrix}
CB & 0 & 0 & \dots  & 0 \\
CB + CAB & CB & 0 & \dots & 0 \\
CB+CAB+CA^2B & CB + CAB &  CB  & \dots & 0 \\
\vdots &\vdots &\vdots &  \ddots &\vdots & \\
\displaystyle \sum_{i=0}^{p-1} CA^iB & \displaystyle\sum_{i=0}^{p-2} CA^iB & \displaystyle\sum_{i=0}^{p-3} CA^iB & \dots & \displaystyle \sum_{i=0}^{p-m}CA^iB 
\end{bmatrix}$$


The MPC objective function:

$$ \arg\underset{\Delta U}\min J=(R_{s}-Y_{p})^TQ(R_{s}-Y_{p}) + \Delta U^TR\Delta U$$


$Y_{p}(k+1)=S^XX(k) + S^U_{pre} U(k-1)+S^U \Delta U_m(k)$

The free response matrix:
$F = S^XX(k) + S^U_{pre}U(k-1)$
$Y = F + S^U \Delta U$

substituting to the cost function:
$$\begin{align}
\arg\underset{\Delta U}\min J &= (R_{s}-Y_{p})^TQ(R_{s}-Y_{p}) + \Delta U^TR\Delta U \\
&= (R_{s}-(F+S^U\Delta U))^TQ(R_{s}-(F+S^U\Delta U)) + \Delta U^TR\Delta U
\end{align}$$

differentiating with respect to $\Delta U$ and setting to $0$:
$$\begin{align}
 \dfrac{\partial J}{\partial\Delta U} &= 0 \\
 -2(S^u)^T Q (R_{s}-F-S^U\Delta U) + 2R\Delta U &=0
\end{align}$$

solving for $\Delta U$,
$$\Delta U= ((S^u)^T QS^u+R)^{-1} (S^U)^T Q(R_{s}-F)$$
