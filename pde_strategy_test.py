def compute_minimized_partial_derivatives(requested_derivatives):
    # Dictionary to store already computed derivatives
    computed = {}
    steps = []

    # Function to extract and compute the sequence of derivatives
    def process_derivative(derivative):
        base = 'u'
        for i in range(2, len(derivative), 2):
            var = derivative[i:i+2]  # Extract the variable part, e.g., 'x1', 'x2', 'x3'
            next_derivative = base + var
            if next_derivative not in computed:
                computed[next_derivative] = True
                steps.append((base, var, next_derivative))
            base = next_derivative

    # Process each requested derivative
    for derivative in requested_derivatives:
        process_derivative(derivative)

    return steps

# Example usage
requested_derivatives = ['ux1', 'ux1x1', 'ux1x2', 'ux1x2x3', 'ux2x2', 'ux2x3']
computed_steps = compute_minimized_partial_derivatives(requested_derivatives)

# Print the computed steps
print(computed_steps)
