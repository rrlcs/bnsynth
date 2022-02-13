
def process():
    # Get Argument Parser
    parser = util.make_arg_parser()
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
