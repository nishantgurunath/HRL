import torch


def a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, l2_reg):

    """update critic"""
    values_pred = value_net(states)
    value_loss = (values_pred - returns).pow(2).mean()
    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()
    #print ("value:", torch.sum(torch.isnan(value_net(states))*1.0))

    """update policy"""
    #print (states.shape)
    log_probs = policy_net.get_log_prob(states, actions)
    policy_loss = -(log_probs * advantages).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()

    #print ("policy:", torch.sum(torch.isnan(policy_net.get_log_prob(states, actions))*1.0))

    return policy_loss, value_loss
