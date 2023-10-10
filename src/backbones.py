import flax.linen as nn
import flax
import jax
import jax.numpy as jnp
from functools import partial
from typing import Any, Callable, Sequence, Tuple, Union

'''
Neural Networks for experiments.
'''

class MLP(nn.Module):
   layer_sizes : Sequence[int]
   dropout : float

   @nn.compact
   def __call__(self, x, training=True):
       activation = x
       for (i, ls) in enumerate(self.layer_sizes[:-1]):
           activation = nn.Dense(ls)(activation)
           activation = nn.Dropout(rate=self.dropout,
                                   deterministic=not training)(activation)
           activation = nn.relu(activation)
       activation = nn.Dense(self.layer_sizes[-1])(activation)
       return activation


class CNN(nn.Module):
    """A simple CNN model."""
    dense1: int = 256 # size of dense layer
    dense2 : int = 256 # size of output layer

    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=self.dense1)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.dense2)(x)
        return x


class MLPHead(nn.Module):
    layer_sizes : Sequence[int]
    backbone : nn.Module

    @nn.compact
    def __call__(self, x, training=True):
        activation = self.backbone(x, training=training)
        activation = nn.relu(activation)
        for (i, ls) in enumerate(self.layer_sizes[:-1]):
           activation = nn.Dense(ls)(activation)
           activation = nn.relu(activation)
        activation = nn.Dense(self.layer_sizes[-1])(activation)
        return activation

class LinearHead(nn.Module):
    backbone : nn.Module
    @nn.compact
    def __call__(self, x, training=True):
        activation = self.backbone(x, training=training)
        activation = nn.relu(activation)
        activation = nn.Dense(10)(activation)
        return activation






ModuleDef = Any

class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


"""
https://github.com/google/flax/blob/main/examples/imagenet/models.py
"""


ModuleDef = Any


class ResNetBlock(nn.Module):
  """ResNet block."""

  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(
      self,
      x,
  ):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(
          self.filters, (1, 1), self.strides, name='conv_proj'
      )(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""

  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(
          self.filters * 4, (1, 1), self.strides, name='conv_proj'
      )(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""

  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  embedding_dim : int = 256
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv

  @nn.compact
  def __call__(self, x, training: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = partial(
        nn.BatchNorm,
        use_running_average=not training,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
    )

    x = conv(
        self.num_filters,
        (3, 3),
        (1, 1),
        padding=[(3, 3), (3, 3)],
        name='conv_init',
    )(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            self.num_filters * 2**i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act,
        )(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.embedding_dim, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x


ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)

