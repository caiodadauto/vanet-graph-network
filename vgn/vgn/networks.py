import sonnet as snt


class FullyLayer(snt.Module):
    def __init__(self,
                 hidden_size,
                 n_layers,
                 activation,
                 norm,
                 dropout_rate,
                 name="FullyLayer"):
        super(FullyLayer, self).__init__(name=name)
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        self._activation = activation

        self._dropout_layers= [snt.Dropout(dropout_rate) for _ in range(self._n_layers)]
        self._norm_layers = [norm() for _ in range(self._n_layers)]
        self._linear_layers = []
        for i in range(self._n_layers):
            factor = .7 + (i / (self._n_layers - 1)) * 0.3
            linear_size = int(np.floor(self._hidden_size * factor))
            self._linear_layers.append(snt.Linear(linear_size))

    def __call__(self, inputs, is_training, activation_kwargs={}, norm_kwargs={}):
        outputs = inputs
        for dropout, linear, norm in zip(self._dropout_layers, self._linear_layers, self._bn_layers):
            outputs = dropout(outputs, is_training)
            outputs = linear(outputs)
            outputs = norm(outputs, **norm_kwargs)
            outputs = self.activation(outputs, **activation_kwargs)
        return outputs


class RecurrentLayer(snt.Module):
    def __init__(self,
                 hidden_size,
                 recurrent,
                 activation,
                 norm,
                 dropout_rate,
                 name="RecurrentLayer"):
        super(RecurrentLayer, self).__init__(name=name)
        self._hidden_size = hidden_size
        self._activation = activation
        self._dropout_cell, self._cell = snt.lstm_with_recurrent_dropout(
            self._hidden_size, keep_prob=dropout_rate)
        self._norm = norm()
        self._recurrent_mask = None

    def get_initial_state(self, batch_size, dtype=tf.float64):
        state, self._recurrent_mask = self._dropout_cell.initial_state(batch_size, dtype=dtype)
        return state

    def __call__(self, inputs, prev_states, is_training, activation_kwargs={}, norm_kwargs={}):
        def true_fn():
            o, h, self._recurrent_mask = self._dropout_gru(inputs, (prev_states, self._recurrent_mask))
            return o, h

        def false_fn():
            o, h = self._gru(inputs, prev_states)
            return o, h

        outputs, next_states = tf.cond(is_training, true_fn=true_fn, false_fn=false_fn)
        outputs = self._norm(outputs, **norm_kwargs)
        outputs = self._activation(outputs, **activation_kwargs)
        return outputs, next_states
