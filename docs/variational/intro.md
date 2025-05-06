# Theory of variational multitask learning and its python API

## Abstract introduction

Variational multitask learning is one of the *multitask approaches* that uses *variational inference* to define probabilistic connections between tasks and do statistical inference on them (a.k.a. learning). This material is a short adaptation of the [original paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/afd4836712c5e77550897e25711e1d96-Paper.pdf)[^1]. It is assummed that the reader is familiar with bayessian statistics and variational inderence[^2]. 

Imagine you have only one task to predict stochastic output $y$ by given input $\mathbf{x}$. Our approach always assumes specific relation between $y$ and $\mathbf{x}$. Namely, it will be *bayessian*: we introduce two hidden random variables. First is *classifier* $\mathbf{w}$ with it's *prior* distribution $p(\mathbf{w})$. Second is  *latent* $\mathbf{z}$ with its prior $p(\mathbf{z} | \mathbf{x})$. The $y$ is then modeled as $p(y | \mathbf{z}, \mathbf{w})$, so there is implicit dependency on input.

The typical situation is that you know your priors up to some parameters. To estimate them from given data $\mathcal{D} = \{ \mathcal{X}, \mathcal{Y}\}$ ($\mathcal{X} = \{\mathbf{x}_j\}_{j=1}^N$, $\mathcal{Y} = \{y_j\}_{j=1}^N$) and to build predictions for new inputs we will use *variational inference*. That means we itroduce *variational distribtions* $q(\mathbf{w} | \mathcal{D})$ and $q(\mathbf{z} | \mathcal{D})$ and maximize *ELBO* objective with respect to var. distributions and priors' parameters. The main method to do it is iterative algorithm called *EM-algorithm*.

Now imagine you have $T$ tasks of the form given above. *Variational multitask* introduce connection between tasks via classifier and latent priors of each task. Namely, the classifer prior of task $i$ is now $p(\mathbf{w}_i | \mathcal{D}_{-i})$ and the latent prior for sample $j$ is $p(\mathbf{z}_{ij} | \mathcal{D}_{-i})$. Here $\mathcal{D}_{-i}$ means samples form all tasks but $i$-th. As one can see this is quite general construction which requires some inital information about priors. To overcome these issues it was proposed to model priors in the following way

$$
\begin{aligned}
    p(\mathbf{w}_{i}|D_{-i}) &= \sum_{k \not= i}\alpha_{ki}q(\mathbf{w}_{i}|D_{k}), \\
    p(\mathbf{z}_{ij} | \mathbf{x}_{ij},D_{-i}) &= \sum_{k \neq i}\beta_{ki}q(\mathbf{z}_{ij} | \mathbf{x}_{ij}, D_{k}),
\end{aligned}
$$

so that prior for task $i$ is a mixture of variational distributions for all other tasks. Mixture coefficients $\alpha_{ki}$ and $\beta_{ki}$ are generated from [*gumbel softmax*](https://en.wikipedia.org/wiki/Gumbel_distribution#Gumbel_reparameterization_tricks), for example

$$
    \alpha_{ki} = \frac{\exp((\log \pi_{ki} + g_{ki})/\tau)}{\sum_{s \neq i} \exp((\log \pi_{si} + g_{si})/\tau)},
$$

where $\tau$ is a temperature parameter, [$g_{ki} \sim Gumbel(0, 1)$](https://en.wikipedia.org/wiki/Gumbel_distribution), $\pi_{ki}$ are learnable parameters that represent "connection strength" between task $k$ and task $i$. Note, this connection is not symmetrical.

Overall, empirical objective to minimize is sampled ELBO:

$$
\begin{align*}
\hat{\mathcal{L}}_{\text{VMTL}}(\theta, \phi, \alpha, \beta) 
&= \frac{1}{T} \sum_{t=1}^{T} \Bigg\{ 
   \sum_{n=1}^{N_t} \Bigg\{ 
   \frac{1}{ML} \sum_{\ell=1}^{L} \sum_{m=1}^{M} \Big[ 
   -\log p(\mathbf{y}_t|\mathbf{z}_{t,n}^{(\ell)}, \mathbf{w}_t^{(m)}) 
   \Big] \\
   &\quad + \mathbb{D}_{\text{KL}} \Big[ 
   q_\phi(\mathbf{z}_{t,n}|\mathbf{x}_{t,n}) \,\Big|\Big|\, 
   \sum_{i \neq t} \beta_{ti} q_\phi(\mathbf{z}_{t,n}|\mathbf{x}_{t,n}, \mathcal{D}_i) 
   \Big] \Bigg\} \\
   &\quad + \mathbb{D}_{\text{KL}} \Big[
   q_\theta(\mathbf{w}_t|\mathcal{D}_i) \,\Big|\Big|\,
   \sum_{i \neq t} \alpha_{ti} q_\theta(\mathbf{w}_i|\mathcal{D}_i)
   \Big] \Bigg\}.
\end{align*}
$$

Here we have introduced parameters $\theta, \phi$ of the variational distributions; all latents are sampled accordingly $\mathbf{z}_{t,n}\sim q_{\phi}(\mathbf{z}_{t,n}|\mathbf{x}_{t,n})$ and $\mathbf{w}_{t} \sim q_{\theta}(\mathbf{w}_{t}|\mathcal{D}_{t})$. The loss structure is intuitively a sum of individual tasks' loss and flexible penalty for tasks disconnection. Due to learnable $\alpha, \beta$ each task can connect to the most relevant neighbours.

After learning procedure, one can obtain predictions using learnt variational distributions:

$$
\begin{align}
    p(\mathbf{y}_t | \mathbf{x}_t) \approx \frac{1}{ML} \sum_{l=1}^L \sum_{m=1}^M p(\mathbf{y}_t | \mathbf{z}_t^{(l)}, \mathbf{w}_t^{(m)}), 
\end{align}\label{eq:pred}
$$

where we draw samples $\mathbf{z}_t^{(l)} \sim q_\phi(\mathbf{z}_t|\mathbf{x}_t)$ and 
$\mathbf{w}_t^{(m)} \sim q_\theta(\mathbf{w}_t|\mathcal{D}_t)$. Given results follow from the general variational inference theory.

## Python API

[`variational`](reference.md) subpackage of the [`bmm_multitask_learning`]() directly follows the theoretical concept of variational multitask learning. Implementaion is based on *pytorch* framework and its [`distribution`](https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution) package.

The `elbo` module stores $\alpha, \beta$ parameters and implements $\hat{\mathcal{L}}_{\text{VMTL}}$ objective. User is required to make and pass `Callable`s with conditional distributions $p(y | \mathbf{z}, \mathbf{w})$, $q(\mathbf{z} | x)$ and unconditional $q(w)$ as [`Distriution`](https://pytorch.org/docs/stable/distributions.html#distribution) objects. User must also provide number of samples for each tasks to properly compute *batched* objective. User can also control temperature sheduling, number of latent samples used in empirical objective, and number of samples used in KL estimation (used if there is no explicit formula for given distributions).

The `distr` module implements prediction distribution (1) and $\mathbb{D}_{\text{KL}}(\cdot, \cdot)$ estimation via sampling. 

## Elemtary example

Go to the simple [demostration](elementary.ipynb) of how `variational` subpackage can be used.

[^1]:
    Shen, Jiayi, et al. "Variational multi-task learning with gumbel-softmax priors." Advances in Neural Information Processing Systems 34 (2021): 21031-21042.

[^2]: Bishop, Christopher M. Pattern Recognition and Machine Learning. New York :Springer, 2006.