from models import (
        HybridNoPEForCausalLM,
        HybridForCausalLM,
        HybridNoPEConfig,
        HybridConfig
    )

def get_model(args, tokenizer):
    if not args.nope:
        config = HybridConfig(
            layers=args.layers,
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size*4,
            num_attention_heads=args.heads,
            d_model=args.hidden_size,
            state_size=args.state_dim,
            ssm_cfg={"d_state": args.state_dim},
            vocab_size=len(tokenizer),
        )
    if args.nope:
        config = HybridNoPEConfig(
            layers=args.layers,
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size*4,
            num_attention_heads=args.heads,
            d_model=args.hidden_size,
            state_size=args.state_dim,
            ssm_cfg={"d_state": args.state_dim},
            vocab_size=len(tokenizer),
        )

    if not args.nope:
        model = HybridForCausalLM(config)
    if args.nope:
        model = HybridNoPEForCausalLM(config)
        
    return model


