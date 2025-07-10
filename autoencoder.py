
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import optax

# Detect available devices (GPUs)
devices = jax.devices()
n_devices = len(devices)
print(f"Running on {n_devices} devices: {devices}")

epochs = 1000
LR     = 0.001

# ========================
# STEP 1: Data & PCA
# ========================
np.random.seed(0)
#N, D, d = 200, 5, 2
d=2
# Assume your data `R_iX` is already loaded here as numpy array of shape (N, D)
R_iX_centered = R_iX - R_iX.mean(axis=0)

pca   = PCA(n_components=d)
pca.fit(R_iX_centered)
V     = pca.components_.T.astype(np.float32)
U_pca = pca.transform(R_iX_centered)

# ========================
# STEP 2: PCA Autoencoder FLAX
# ========================
class autoencoder(nn.Module):
    d    : int
    D    : int
    bias : bool = False

    @nn.compact
    def __call__(self, x):
        z = nn.Dense(self.d, use_bias=self.bias, name='encoder')(x)
        y = nn.Dense(self.D, use_bias=self.bias, name='decoder')(z)
        return y, z

model = autoencoder(d=d, D=D)

# ========================
# STEP 3: Initialization + Noise
# ========================
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, D)))

# Slight noise for initialization
encoder_noise = 0.001 * jax.random.normal(rng, shape=V.shape)
decoder_noise = 0.001 * jax.random.normal(rng, shape=V.T.shape)

params = freeze({
    'params': {
        'encoder': {'kernel': jnp.array(V)   + encoder_noise},
        'decoder': {'kernel': jnp.array(V.T) + decoder_noise}
    }
})

# ========================
# STEP 4: OPTAX optimizer setup (multi-device)
# ========================
optimizer = optax.adam(LR)
opt_state = optimizer.init(params)

# Replicate params and opt_state across devices
params_repl = jax.device_put_replicated(params, devices)
opt_state_repl = jax.device_put_replicated(opt_state, devices)

# ========================
# STEP 5: Prepare dataset for multi-GPU
# ========================
# Split data into n_devices chunks along batch dimension
assert N % n_devices == 0, "N must be divisible by the number of devices!"
batch_per_device = N // n_devices
data_sharded = R_iX_centered.reshape(n_devices, batch_per_device, D)

# ========================
# STEP 6: Define pmap training step
# ========================
def mse_loss(params, batch):
    recon, _ = model.apply(params, batch)
    return jnp.mean((batch - recon)**2)

#@jax.pmap(axis_name='batch')  # ← ADD THIS
def train_step_p(params, opt_state, batch):
    loss, grads = jax.value_and_grad(mse_loss)(params, batch)
    grads = jax.lax.pmean(grads, axis_name='batch')
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    loss = jax.lax.pmean(loss, axis_name='batch')
    return params, opt_state, loss

train_step = jax.pmap(train_step_p, axis_name='batch')

# ========================
# STEP 7: Multi-GPU Training Loop
# ========================
losses = []
for epoch in range(epochs):
    params_repl, opt_state_repl, loss = train_step(params_repl, opt_state_repl, data_sharded)
    loss_val = loss.mean()
    losses.append(loss_val.item())
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss_val:.6f}")

losses = np.array(losses)

# ========================
# STEP 8: Embeddings after training
# ========================
# Collect params from device 0 (all are identical after pmean)
trained_params = jax.tree_map(lambda x: x[0], params_repl)

_, embeddings_trained = model.apply(trained_params, R_iX_centered)

# ========================
# STEP 9: Visualization
# ========================
fig, axs = plt.subplots(1, 1, figsize=(5, 5))

sns.scatterplot(x=U_pca[:,0], y=U_pca[:,1], ax=axs, label="Original sklearn PCA Embedding")
axs.set_title('PCA Embedding')
sns.scatterplot(x=embeddings_trained[:,0], y=embeddings_trained[:,1], ax=axs, color='green', label="Embedding After Multi-GPU Training")
plt.tight_layout()
plt.show()

# Loss Curve
plt.figure(figsize=(8,4))
plt.plot(losses - losses[-1])
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve (Multi-GPU)")
plt.grid()
plt.show()

# ========================
# STEP 10: Numerical Comparison
# ========================
print("Mean squared difference between sklearn PCA and multi-GPU AE embeddings:",
      np.mean((U_pca - embeddings_trained)**2))
