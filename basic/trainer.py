import tensorflow as tf

from basic.model import Model
from my.tensorflow import average_gradients


class Trainer(object):
    def __init__(self, config, model):
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.AdadeltaOptimizer(config.init_lr)
        self.loss = model.get_loss()
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch, get_summary=False):
        assert isinstance(sess, tf.Session)
        _, ds = batch
        feed_dict = self.model.get_feed_dict(ds, True)
        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op


class MultiGPUTrainer(object):
    def __init__(self, config, models):
        model = models[0]
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.AdadeltaOptimizer(config.init_lr)
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.models = models
        losses = []
        grads_list = []
        for gpu_idx, model in enumerate(models):
            with tf.name_scope("grads_{}".format(gpu_idx)), tf.device("/{}:{}".format(config.device_type, gpu_idx)):
                loss = model.get_loss()
                grads = self.opt.compute_gradients(loss, var_list=self.var_list)
                losses.append(loss)
                grads_list.append(grads)
        self.loss = tf.add_n(losses)/len(losses)
        self.grads = average_gradients(grads_list)
        self.do_accumulate_grads = tf.placeholder('bool', [], name='do_accumulate_grads')
        
        # Initialize and accumulate the gradients and losses
        accumulated_grads = []
        accumulate_grads_ops = []
        with tf.variable_scope("accumulate_grads"):
            for grad, var in self.grads:
                accumulated_grad = tf.Variable(tf.zeros(var.get_shape()),
                                               name=('accumulated_'+grad.name).split(':')[0])
                accumulated_grad = tf.cond(self.do_accumulate_grads, lambda: tf.assign_add(
                   accumulated_grad, grad), lambda: tf.assign(accumulated_grad, grad))
                accumulated_grad_and_var = (accumulated_grad, var)
                accumulated_grads.append(accumulated_grad_and_var)
                accumulate_grads_ops.append(accumulated_grad)
            self.accumulated_grads_op = tf.tuple(accumulate_grads_ops)
        with tf.variable_scope("accumulate_losses"):
            self.accumulated_loss = tf.Variable(tf.zeros([]), name='accumulated_loss')
            self.accumulated_loss = tf.cond(self.do_accumulate_grads, lambda: tf.assign_add(
                self.accumulated_loss, self.loss), lambda: tf.assign(self.accumulated_loss, self.loss))

        # Accumulated and averaged gradients and loss
        with tf.variable_scope("average_accumulated_grads"):
            self.averaged_accumulated_grads = []
            for accumulated_grad, var in accumulated_grads:
                averaged_accumulated_grad_and_var = (
                    accumulated_grad / self.config.num_grad_accumulate_iteration, var)
                self.averaged_accumulated_grads.append(averaged_accumulated_grad_and_var) 
        with tf.variable_scope("average_accumulated_loss"):
            self.averaged_accumulated_loss = self.accumulated_loss / self.config.num_grad_accumulate_iteration
        tf.scalar_summary("model_0/loss", self.averaged_accumulated_loss)
        self.train_summary = tf.merge_all_summaries()
        self.train_summary = tf.merge_summary(tf.get_collection("summaries", self.model.scope))
        # Training operation with accumulated gradients
        self.train_op = self.opt.apply_gradients(self.averaged_accumulated_grads,
                                                 global_step=self.global_step)

    def accum_grad_step(self, sess, batches, do_accumulate_grads):
        assert isinstance(sess, tf.Session)
        feed_dict = {}
        for batch, model in zip(batches, self.models):
            _, ds = batch
            feed_dict.update(model.get_feed_dict(ds, True))
        feed_dict.update({self.do_accumulate_grads: do_accumulate_grads}) 

        # Forward-backward graph and accumulate loss and gradients
        sess.run([self.accumulated_loss, self.accumulated_grads_op], feed_dict=feed_dict)

    def step(self, sess, batches, get_summary=False, do_accumulate_grads=True):
        assert isinstance(sess, tf.Session)
        feed_dict = {}
        for batch, model in zip(batches, self.models):
            _, ds = batch
            feed_dict.update(model.get_feed_dict(ds, True))
        feed_dict.update({self.do_accumulate_grads: do_accumulate_grads}) 

        if get_summary:
            loss, summary, train_op = \
            sess.run([self.averaged_accumulated_loss, self.train_summary, self.train_op],
                     feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.averaged_accumulated_loss, self.train_op],
                                      feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op
