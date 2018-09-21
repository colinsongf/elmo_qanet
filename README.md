# Three-attention QANet with elmo
This is a tensorflow implementation of three attention with elmo model. The input embedding layer is similar to the one in the QANet model except for our adding elmo part. As for the embedding encoding layer, the same one in QANet is used. And we also have the similar layer like context-query attention layer. Candidate answers information are added. We calculated the bi-attention between query and context first and then calculate the bi-attention between candidate answers and query-context. Finally, we only have an output layer. After getting three-attention vectors, we multiply them by transformed candidates matrix. And corss entropy loss between predicted vector and ground true label vector is used. In the testing, we select the one which has maximum probability as the answer.




