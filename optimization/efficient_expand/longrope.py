"""py module for implementing the LongRoPE from microsoft research
LongRoPE = YaRN + StreamingLLM + Optimization Problem

λ(theta) is the variable for grouping the RoPE hidden state dimension
n^ is the variable for starting tokens, motivated by StreamingLLM from MiT
LongRoPE search the optimal param value of theta, n
"""

