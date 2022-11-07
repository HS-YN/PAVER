from torch.optim.lr_scheduler import LambdaLR

from exp import ex


def get_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            maxval = float(num_training_steps - current_step)
            maxval /= float(max(1., num_training_steps - num_warmup_steps))
            return max(0.0, maxval)

    return LambdaLR(optimizer, lr_lambda)


def get_no_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        return 1

    return LambdaLR(optimizer, lr_lambda)



sched_dict = {
    'warmup': get_warmup_scheduler,
    'none': get_no_scheduler
}


@ex.capture()
def get_scheduler(optimizer, t_total, max_epoch, warmup_epoch, scheduler_name):
    return sched_dict[scheduler_name](optimizer, int(t_total * warmup_epoch / max_epoch), t_total)
    