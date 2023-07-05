$Q(s, a) \leftarrow (1 - \alpha) \cdot Q(s, a) + \alpha \cdot \left(r + \gamma \cdot \max_{a'} Q(s', a')\right)$

Independent Q

$Q^i_{t+1}(s, a) \leftarrow (1 - \alpha) \cdot Q^i_t(s, a) + \alpha \cdot \left(r_i + \gamma \cdot \max_{a'} Q^i_t(s', a')\right)$

$where\ i=1, 2, ..., N$

<br/>

Share Q

$\bar Q_{t}(s_i, a_i) \leftarrow (1 - \alpha) \cdot \bar Q_t(s_i, a_i) + \alpha \cdot \left(r_i + \gamma \cdot \max_{a_i'} \bar Q_t(s_i', a_i')\right)$

$where\ i=1, 2, ..., N$

<br/>

$For\ i\ in\ (1,2,...,N)$

$Q^i_t(s, a) \leftarrow \bar Q_t(s, a)$

$Q^i_{t}(s, a) \leftarrow (1 - \alpha) \cdot Q^i_t(s, a) + \alpha \cdot \left(R(s,a) + \gamma \cdot \max_{a'} Q^i_t(s', a')\right)$

$\bar Q_t(s, a) \leftarrow Q^i_t(s, a)$

$End\ For$

<br/>

Q Avg

\# Clients

$For\ i\ in\ (1,2,...,N)$

$Q^i_t \leftarrow \bar Q_0$

$For\ j\ in\ NumLocalEpisode:$

```
$Q^i_{t}(s, a) \leftarrow (1 - \alpha) \cdot Q^i_t(s, a) + \alpha \cdot \left(r + \gamma \cdot \max_{a'} Q^i_t(s', a')\right)$
```

$End\ For$

$End\ For$

\# Server

$\bar Q_t \leftarrow \frac{1}{n}\sum^n_{i=1}Q^i_t$   # QAvg

<br/>

Q Avg Diff

\# Clients

$For\ i\ in\ (1,2,...,N)$

$Q^i_t \leftarrow \bar Q_0$

$For\ j\ in\ NumLocalEpoch:$

```
$Q^i_{t}(s, a) \leftarrow (1 - \alpha) \cdot Q^i_t(s, a) + \alpha \cdot \left(r + \gamma \cdot \max_{a'} Q^i_t(s', a')\right)$
```

$End\ For$

$\Delta Q^i_t=Q^i_t-\bar Q_0$

$End\ For$

\# Server

$\bar Q_t \leftarrow \bar Q_0+\frac{1}{n}\sum^n_{i=1} \Delta Q^i_t$   # Q Avg Delta

<br/>

Q Avg Max

\# Clients

$For\ i\ in\ (1,2,...,N)$

$Q^i_t \leftarrow \bar Q_0$

$For\ j\ in\ NumLocalEpoch:$

```
$Q^i_{t}(s, a) \leftarrow (1 - \alpha) \cdot Q^i_t(s, a) + \alpha \cdot \left(r + \gamma \cdot \max_{a'} Q^i_t(s', a')\right)$
```

$End\ For$

$\Delta Q^i_t=Q^i_t-\bar Q_0$

$End\ For$

\# Server

$Idx \leftarrow argmax_i|\Delta Q_i(s,a)|$  # Q Max Delta

$\Delta Q_{max} \leftarrow \Delta Q_i[Idx]$

$\bar Q_t \leftarrow \bar Q_0+\Delta Q_{max}$ 

<br/>

Q Avg All

\# Clients

$For\ i\ in\ (1,2,...,N)$

```
$Q^i_t \leftarrow \bar Q_0$

$For\ j\ in\ NumLocalEpoch:$

    $Q^i_{t}(s, a) \leftarrow (1 - \alpha) \cdot Q^i_t(s, a) + \alpha \cdot \left(r + \gamma \cdot \max_{a'} Q^i_t(s', a')\right)$

$End\ For$

$\Delta Q^i_t=Q^i_t-\bar Q_0$
```

$End\ For$

\# Server

$\bar Q_t \leftarrow \bar Q_0+\Delta Q^i_t$   # Q All

<br/>

Q Avg All

\# Clients

$For\ i\ in\ (1,2,...,N)$

```
$Q^i_t \leftarrow \bar Q_0$

$For\ j\ in\ NumLocalEpoch:$

    $Q^i_{t}(s, a) \leftarrow (1 - \alpha) \cdot Q^i_t(s, a) + \alpha \cdot \left(r + \gamma \cdot \max_{a'} Q^i_t(s', a')\right)$

$End\ For$

$\Delta Q^i_t=Q^i_t-\bar Q_0$
```

$End\ For$

\# Server
\epsilon = max(\frac{1}{n}, \epsilon \cdot decay_factor)

$\bar Q_t \leftarrow \bar Q_0+ \epsilon \cdot \Delta Q^i_t$   # Q All

<br/>

Q Avg Template

\# Clients

$For\ i\ in\ (1,2,...,N)$

$Q^i_t \leftarrow \bar Q_0$

$For\ j\ in\ NumLocalEpoch:$

$Q^i_{t}(s, a) \leftarrow (1 - \alpha) \cdot Q^i_t(s, a) + \alpha \cdot \left(r + \gamma \cdot \max_{a'} Q^i_t(s', a')\right)$

$End\ For$

$\Delta Q^i_t=Q^i_t-\bar Q_0$

$End\ For$

\# Server

$\bar Q_t \leftarrow \bar Q_0+\frac{1}{n}\sum^n_{i=1} \Delta Q^i_t$   # QAll

<br/>

$Q^i_{t+1}(s, a) \leftarrow (1 - \alpha) \cdot Q^i_t(s, a) + \alpha \cdot \left(R(s,a) + \gamma \cdot \max_{a'} Q^i_t(s', a')\right)$

$After\ all\ client\ update\ Q_k$

$\bar Q_t(s, a) \leftarrow \frac{1}{n}\sum^n_{i=1}Q^i_t(s, a)$

$Q_i(s_t, a_t) \leftarrow \bar Q(s_t, a_t)$

<br/>

$After UpdateQ_i$

$\Delta Q_i \leftarrow Q_i -\bar Q$

$Idx \leftarrow argmax_i|\Delta Q_i|$

$\Delta Q_{max} \leftarrow \Delta Q_i[Idx]$

$\bar Q \leftarrow \bar Q + \Delta Q_{max}$

$After UpdateQ_i$

$\Delta Q_i \leftarrow Q_i -\bar Q$

$\Delta Q_{avg} \leftarrow \frac{1}{n}\sum^n_{i=1}\Delta Q_i$

$\bar Q \leftarrow \bar Q + \Delta Q_{avg}$

$Q_i(s_t, a_t) \leftarrow \bar Q(s_t, a_t)$

$After UpdateQ_i$

$\Delta Q_i \leftarrow Q_i -\bar Q$

$\bar Q \leftarrow \bar Q + \sum_i \Delta Q_{i}$

$Q_i \leftarrow \bar Q$

<br/>

$$MAE=\frac{1}{n}\sum^n_j|Q_j^t-Q_j^{t+1}|$$
