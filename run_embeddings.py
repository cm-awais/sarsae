import embeddings as E
MODELS = ['sarjepa', 'saratrx', 'dofa_b', 'dinov2_b', 'clip_b', 'dinov3_b_web', 'dinov3_l_web', 'dinov3_l_sat']
E.extract_all(MODELS, device='cuda')
E.status(MODELS)
