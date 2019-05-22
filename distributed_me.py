import copy
from torch.nn.modules import Module

import torch
import torch.distributed as dist
from torch.cuda._utils import _get_device_index

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group


from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, _flatten_sparse_tensors, _unflatten_sparse_tensors

class DistributedDataParallel(Module):

    def __init__(self, module, device_id=None,
                 output_device=None, dim=0, broadcast_buffers=True,
                 process_group=None, bucket_cap_mb=25,
                 check_reduction=False, sparse_ratio = 0.1, sparse_threshold = 1024):

        super(DistributedDataParallel, self).__init__()
        self.module = module

        if device_id is None:
            raise RuntimeError("device_id cannot be None")

        if output_device is None:
            output_device = device_id

        if process_group is None:
            self.process_group = _get_default_group()
        else:
            self.process_group = process_group

        self.dim = dim
        self.module = module
        self.device_id = _get_device_index(device_id, True)
        self.output_device = _get_device_index(output_device, True)
        self.broadcast_buffers = broadcast_buffers
        self.check_reduction = check_reduction
        self.sparse_ratio = sparse_ratio
        self.sparse_threshold = sparse_threshold

        MB = 1024 * 1024

        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = 250 * MB

        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._dist_broadcast_coalesced(module_states,
                                           self.broadcast_bucket_size)

        self._ddp_init_helper()

    def _ddp_init_helper(self):

        self.modules_params_data = [p.data for p in self.module.parameters()]
        self.modules_buffers_data = [b.data for b in self.module.buffers()]

        self.buckets = []
        self.flat_parameter = {}

        self.bucket_map = {}
        param_buckets = []
        self.bucket_sizes = []
        self.parameter_length = 0
        self.offset_map = {}
        self.sparse_length ={}
        for p in self.module.parameters():
            if p.requires_grad:
                param_buckets.append(p)
                self.parameter_length += 1


        length_temp = 0
        self.sparse_length_all = 0
        self.length_all = 0
        for bucket_idx, p in enumerate(param_buckets):
            self.bucket_sizes.append(0)
            self.bucket_map[p] = bucket_idx
            self.offset_map[p] = length_temp
            self.flat_parameter[p] = _flatten_dense_tensors(p)
            length_temp += self.flat_parameter[p].shape[0]
            self.length_all += self.flat_parameter[p].shape[0]
            self.sparse_length[p] = max(int(_flatten_dense_tensors(p).shape[0]*self.sparse_ratio),min(self.flat_parameter[p].shape[0],self.sparse_threshold))
            self.sparse_length_all += self.sparse_length[p]
            self.bucket_sizes[bucket_idx] += 1


        self.buckets = [None]
        self.buckets_ready_size = [0 for i in range(len(self.bucket_sizes))]

        self.buckets_coalesced = [[] for _ in range(len(self.bucket_sizes))]

        self.next_bucket = len(self.bucket_sizes) - 1


        self.all_buckets_reduced = False
        self.check_previous_reduction = False


        self.reduction_works = [None for _ in range(len(self.bucket_sizes))]
        self.devs_ready = [0 for _ in range(len(self.bucket_sizes))]

        self._register_grad_hooks()

    def _dist_broadcast_coalesced(self, tensors, buffer_size):
        dist._dist_broadcast_coalesced(self.process_group, tensors, buffer_size, False)

    def __getstate__(self):
        self._check_default_group()
        attrs = copy.copy(self.__dict__)
        del attrs['process_group'], \
            attrs['default_streams'], \
            attrs['_grad_accs']
        return attrs

    def __setstate__(self, state):
        # If serializable, then the process group should be the default one
        self.process_group = _get_default_group()
        self.check_previous_reduction = False
        super(DistributedDataParallel, self).__setstate__(state)
        self._ddp_init_helper()


    def forward(self, *inputs, **kwargs):

        #print('This is the forward')
        if self.check_reduction:
            self._check_previous_reduction()
        #print('This is the forward 1')
        self._sync_params()
        outputs = self.module(*inputs, **kwargs)
        #print('This is the forward 2')
        return outputs


    def _sync_params(self):



        # module buffer sync
        if self.broadcast_buffers:
            if len(self.modules_buffers_data) > 0:
                # cross-node buffer sync
                self._dist_broadcast_coalesced(self.modules_buffers_data,
                                               self.broadcast_bucket_size)


    def _check_previous_reduction(self):
        if not self.training:
            return
        # self.check_previous_reduction will be False in the first iteration
        # and is then toggled to True for all future iterations.
        if self.check_previous_reduction is False:
            self.check_previous_reduction = True
        else:
            if not self.all_buckets_reduced:
                raise RuntimeError("Not all gradients have been reduced from "
                                   "the backward of the previous iteration. "
                                   "This is unexpected and fatal error. Please "
                                   "check and ensure that the model's "
                                   "parameters are not changed after you wrap "
                                   "up the model with DistributedDataParallel.")
        self.all_buckets_reduced = False

    def _register_grad_hooks(self):
        self._grad_accs = []  # need to keep them in scope

        # default stream tracking to launch nccl reduce kernels
        self.default_streams = []
        with torch.cuda.device(self.device_id):
            self.default_streams.append(torch.cuda.current_stream())

        for p in self.module.parameters():
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_param_hook(p))
                self._grad_accs.append(grad_acc)

    def train(self, mode=True):
        self.check_previous_reduction = False
        super(DistributedDataParallel, self).train(mode)
        self.module.train(mode)

    def _make_param_hook(self, param):

        def distributed_data_parallel_hook(*unused):
            if param.grad.requires_grad:
                raise RuntimeError("DistributedDataParallel only works "
                                   "with gradients that don't require grad")
            # print('This is the _make_param_hook')





            #self._queue_reduction(bucket_idx)
            self.next_bucket -= 1

            if self.next_bucket == -1:
                # A final sync for all the reduction works
                self._sync_reduction_works()
                self.all_buckets_reduced = True




        return distributed_data_parallel_hook




    def _sync_reduction_works(self):
        # Now only work on the first GPU of self.device_ids
        # _sync_reduction will use a seperate CUDA stream to uncoalesce
        # the coalesced tensors to achieve more parallelisms
        temp = [None for _ in range(self.parameter_length)]
        for p in self.module.parameters():
            if p.requires_grad:
                bucket_idx = self.bucket_map[p]
                temp[bucket_idx] = p.grad.data
        flatten_tensor = _flatten_dense_tensors(temp)
        self.buckets = flatten_tensor/self.process_group.size()
        dist.all_reduce(self.buckets,async_op=False)
        dense_tensor = _unflatten_dense_tensors(self.buckets,temp)

        for p in self.module.parameters():
            if p.requires_grad:
                bucket_idx = self.bucket_map[p]
                p.grad.data.copy_(dense_tensor[bucket_idx])


        # Reset the module states
        self.next_bucket = len(self.bucket_sizes) - 1
        self.reduction_works = [None for _ in range(len(self.bucket_sizes))]
        self.devs_ready = [0 for _ in range(len(self.bucket_sizes))]

        self.buckets = [None]
        self.buckets_coalesced = [[] for _ in range(len(self.bucket_sizes))]
        self.buckets_ready_size = [0 for i in range(len(self.bucket_sizes))]