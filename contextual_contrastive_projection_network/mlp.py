from contextual_contrastive_projection_network.hyperparameters import HyperParameters
import torch

class ContrastiveMLP(torch.nn.Module):

    def __init__(self, args: HyperParameters):

        super(ContrastiveMLP, self).__init__()
        input_size = args.input_layer
        activation = self.__build_activation(args)
        gate = Gate(input_size) if args.enable_gate else torch.nn.Identity()
        noise = self.__build_noise(args)
        middle_layer = self.__build_middle_layer(input_size, args)
        output_layer = self.__build_output_layer(input_size, args)

        self.net = torch.nn.Sequential(
            gate,
            middle_layer,
            activation,
            noise,
            output_layer,
            # torch.nn.LayerNorm(args.output_layer)
        )

        print("created mlp model: ", self.net)

    def forward(self, x):
        return self.net(x)


    def __build_activation(self, args: HyperParameters):
        if args.activation == "silu":
            return torch.nn.SiLU()
        elif args.activation == "leaky_relu":
            return torch.nn.LeakyReLU()
        return torch.nn.ReLU()

    def __build_noise(self, args: HyperParameters):
        if args.noise == "dropout":
            return torch.nn.Dropout(args.dropout)
        return torch.nn.Identity()

    def __build_middle_layer(self, input_layer:int, args: HyperParameters):
        if args.is_hidden_layer:
            return torch.nn.Linear(input_layer, args.hidden_layer)
        return torch.nn.Identity()

    def __build_output_layer(self, input_layer:int,  args: HyperParameters):
        if args.is_hidden_layer:
            return torch.nn.Linear(args.hidden_layer, args.output_layer)
        return torch.nn.Linear(input_layer, args.output_layer)

class Gate(torch.nn.Module):
    def __init__(self, size):
        super(Gate, self).__init__()
        self.dimension = size
        self.gate = torch.nn.Parameter(torch.ones(size))

    def forward(self, x):
        neuron_to_enable = torch.sigmoid(self.gate)
        return x * neuron_to_enable

    def extra_repr(self) -> str:
        return f"dimension={self.dimension}"