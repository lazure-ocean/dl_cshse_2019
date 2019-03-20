def get_generator(input_tensor, hidden, hidden_size, generator, max_length=MAX_LENGTH):
    input_length = input_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, hidden_size, device=device)
    
    for ind, element in enumerate(input_tensor):
                output, hidden = generator[0](element, hidden)
                encoder_outputs[ind] = output.view(hidden_size)
    
    decoder_outputs = torch.zeros(max_length, hidden_size, device=device)
    for ind, elementin enumerate(encoder_outputs):
        output , hidden , attention = generator[1](element, hidden, encoder_outputs)
        decoder_outputs[ind] = output

    return decoder_outputs

def get_discriminator(input_tensor, hidden, hidden_size, discriminator, max_length=MAX_LENGTH):
    input_length = input_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, hidden_size, device=device)
    
    for ind, element in enumerate(input_tensor):
                output, hidden = discriminator[0](element, hidden)
                encoder_outputs[ind] = output.view(hidden_size)
    
    decoder_outputs = torch.zeros(max_length, hidden_size, device=device)
    for ind, elementin enumerate(encoder_outputs):
        output , hidden , attention = discriminator[1](element, hidden, encoder_outputs)
        decoder_outputs[ind] = output
        
    out = discriminator[4](decoder_outputs)

    return out

def create_critic_loss(cumulative_rewards, estimated_values, missing):
    """Compute Critic loss in estimating the value function.  This should be an
    estimate only for the missing elements."""
    
    loss = nn.MSELoss(cumulative_rewards * missing, estimated_values * missing)
    return loss

def get_critic(generator_outputs, dicscriminator_outputs, input_tensor, target_tensor, gamma=0.9, baseline_method='critic'):
    eps = 1e-7
    input_length = input_tensor.size()[0]
    missing = torch.tensor([1 if a != b else 0 for (a, b) in zip(input_tensor, target_tensor)])
    estimated_values = discriminator_outputs
    discriminator_outputs = nn.LogSigmoid(discriminator_outputs)
    rewards = torch.log(dis_predictions + eps)
    rewards_list = rewards.view(reward.size()[0]) * missing
    log_probs_list = generator_outputs.view(generator_outputs.size()[0]) * missing
    
    cumulative_rewards = []
    for t in xrange(FLAGS.sequence_length):
        cum_value = tf.zeros(shape=[FLAGS.batch_size])
        for s in xrange(t, FLAGS.sequence_length):
            cum_value += missing[s] * np.power(gamma, (s - t)) * rewards_list[s]
        cumulative_rewards.append(cum_value)
    cumulative_rewards = torch.tensor(cumulative_rewards)
    
    if baseline_method not None:

        # Critic loss calculated from the estimated value function \hat{V}(s)
        # versus the true value function V*(s).
        critic_loss = create_critic_loss(cumulative_rewards, estimated_values, missing)

        # Baselines are coming from the critic's estimated state values.
        baselines = estimated_values.view(estimated_values.size()[0])

        ## Calculate the Advantages, A(s,a) = Q(s,a) - \hat{V}(s).
        advantages = []
        for t in xrange(input_length):
            log_probability = log_probs_list[t]
            cum_advantage = torch.zeros(input_length, requires_grad=True)

            for s in xrange(t, input_length):
                cum_advantage += missing_list[s] * np.power(gamma, (s - t)) * rewards_list[s]
            cum_advantage -= baselines[t]
            advantages.append(missing_list[t] * cum_advantage)
            final_gen_objective += torch.mm(
              log_probability, missing_list[t] * cum_advantage.grad)
    
    return final_gen_objective
 
def train(input_tensor, target_tensor, generator, discriminator, lang, criterion,  critic_iter, gen_train_iter, dis_train_iter, dataset, n=10 ,max_length=MAX_LENGTH, gamma=0.9):
    for iteration in range(critic_iter):
        generator_loss = 0
        for sentence_number in range(gen_train_iter):
            hidden = generator[0].initHidden()
            hidden_size = generator[0].hidden_size
            generator[2].zero_grad()
            generator[3].zero_grad()

            generator_outputs = get_generator(input_tensor, hidden, hidden_size, generator)
            
            discriminator[2].zero_grad()
            discriminator[3].zero_grad()
            
            discriminator_outputs = get_discriminator(generator_outputs, hidden, hidden_size, discriminator)
            
            generator_loss += get_critic(generator_outputs, discriminator_outputs, input_tensor, target_tensor)
            
        generator_loss.backward()
    
    for iteraion in range(dis_train_iter):
        fake_indexes = np.random.choice(dataset.size()[0], n, replace=False)
        real_indexes = np.random.choice(dataset.size()[0], n, replace=False)
        fake_samples = dataset[fake_indexes]
        real_samples = dataset[real_indexes]
        fake_predictions = tensor.zeroes(n, max_length)
        real_predictions = tensor.zeroes(n, max_length)
        
        for ind, sample in enumerate(fake_samples):
            sample = sample * generate_mask()
            fake_predictions[ind] = get_generator(sample, hidden, hidden_size, generator)
            
        for ind, sample in enumerate(real_samples):
            real_predictions[ind] = get_generator(sample, hidden, hidden_size, generator)
        