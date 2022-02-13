import code.preprocessor as preprocessor

if __name__ == "__main__":

    # Get Argument Parser
    parser = util.make_arg_parser()
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if preprocessor = 1:
        #Manthan 1 code
    else:
        #Manthan 2 code











    # 1. Preprocess input data

    samples = preprocessor.process()

    # 2. Feed samples into GCLN

    skolem_function = model()

    # 3. Postprocess skolem function from GCLN
