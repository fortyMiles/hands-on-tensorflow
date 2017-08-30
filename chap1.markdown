### Chapter 1 Exercises

#### 1. What are the main benefits of creating a computation graph rather than directly executing the computations? What are the main drawbacks?

Answer:

Benefits: 1. Could split into several servers or gpus. This will make the distributed computing and
multiply GPU computing easier; 2. Could make the auto-diff easier, which could speed up the computation.

Drawbacks: 1. It some difficult to create or understand well to define a computation graph.
2. hardly build computation dynamic; 3. debug is difficult.

#### 2. If you create a graph g containing a variable w, then start two threads and open a session in each thread, both using the same graph g, will each session have its own copy of the variable w or will it be shared?

Answer: copy.


#### 3. When is a variable initialized? When is it destroyed?

Answer:

initialized when we defined init = tf.initial run. or eval.
destroyed when session is end.


#### 4. What is the difference between a placeholder and a variable?

Answer:

placeholder need feed data each time.

variable could change value by operation.

#### 5. When you run a graph, can you feed the output value of any operation, or just the value of placeholders?

can feed the output value of any operation, not just tht value of placeholders.

But in practice, this is rather rare. But it sometime will be useful.

#### 6. How can you set a variable to any value you want (during the execution phase)?

Have no idea.

```python
    var1 = tf.Variable(0)

    new_var1_value = tf.placeholder(dtype=tf.int32, shape=())

    var1_assign = tf.assign(var1, new_var1_value)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        print(var1.eval())
        var1_assign.eval(feed_dict={new_var1_value: 1})
        print(var1.eval())
 ```

