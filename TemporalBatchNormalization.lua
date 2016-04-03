local BN, parent = torch.class('nn.TemporalBatchNormalization', 'nn.BatchNormalization')

-- expected dimension of input
BN.nDim = 3