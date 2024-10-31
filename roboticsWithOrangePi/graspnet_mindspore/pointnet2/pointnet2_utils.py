
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as P
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import constexpr

# class RandomDropout(nn.Module):
#     def __init__(self, p=0.5, inplace=False):
#         super(RandomDropout, self).__init__()
#         self.p = p
#         self.inplace = inplace

#     def forward(self, X):
#         theta = torch.Tensor(1).uniform_(0, self.p)[0]
#         return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)

@constexpr
def generate_tensor_fps(B, N):
    """generate tensor"""
    farthest = Tensor(np.random.randint(N, size=(B,)), ms.int32)
    return farthest
@constexpr
def generate_tensor_batch_indices(B):
    """generate tensor"""
    return Tensor(np.arange(B), ms.int32)

def furthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, _ = xyz.shape
    centroids = mnp.zeros((npoint, B), ms.int32)
    distance = mnp.ones((B, N), ms.int32) * 1e9
    farthest = generate_tensor_fps(B, N)
    batch_indices = generate_tensor_batch_indices(B)
    for i in range(npoint):
        centroids = P.Cast()(centroids, ms.float32)
        farthest = P.Cast()(farthest, ms.float32)
        centroids[i] = farthest
        centroids = P.Cast()(centroids, ms.int32)
        farthest = P.Cast()(farthest, ms.int32)
        index = P.Concat(-1)((batch_indices.reshape(batch_indices.shape + (1,)),
                              farthest.reshape(farthest.shape + (1,))))
        centroid = P.GatherNd()(xyz, index).reshape((B, 1, 3))
        dist = P.ReduceSum()((xyz - centroid) ** 2, -1)
        distance = P.Minimum()(distance, dist)
        farthest = P.Argmax()(distance)
    return P.Transpose()(centroids, (1, 0))




def gather_operation(features, idx):
    r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """
    return P.GatherNd()(features, idx.unsqueeze(-1).expand(idx.shape + (features.shape[1],)))

def three_nn(unknown, known):
    r"""
        Find the three nearest neighbors of unknown in known
    Parameters
    ----------
    unknown : ms.Tensor
        (B, n, 3) tensor of known features
    known : torch.Tensor
        (B, m, 3) tensor of unknown features

    Returns
    -------
    dist : ms.Tensor
        (B, n, 3) l2 distance to the three nearest neighbors
    idx : torch.Tensor
        (B, n, 3) index of 3 nearest neighbors
    """
    dist2 = square_distance(unknown, known)
    dist2, idx = P.TopK()(dist2, 3)
    return P.Sqrt()(dist2), idx

def three_interpolate(features, idx, weight):  
    r"""
#             Performs weight linear interpolation on 3 features
#         Parameters
#         ----------
#         features : ms.Tensor
#             (B, c, m) Features descriptors to be interpolated from
#         idx : ms.Tensor
#             (B, n, 3) three nearest neighbors of the target features in features
#         weight : ms.Tensor
#             (B, n, 3) weights

#         Returns
#         -------
#         ms.Tensor
#             (B, c, n) tensor of the interpolated features
#         """
    B, c, m = features.shape
    n = idx.shape[1]


# class ThreeInterpolate(Function):
#     @staticmethod
#     def forward(ctx, features, idx, weight):
#         # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
#         r"""
#             Performs weight linear interpolation on 3 features
#         Parameters
#         ----------
#         features : torch.Tensor
#             (B, c, m) Features descriptors to be interpolated from
#         idx : torch.Tensor
#             (B, n, 3) three nearest neighbors of the target features in features
#         weight : torch.Tensor
#             (B, n, 3) weights

#         Returns
#         -------
#         torch.Tensor
#             (B, c, n) tensor of the interpolated features
#         """
#         B, c, m = features.size()
#         n = idx.size(1)

#         ctx.three_interpolate_for_backward = (idx, weight, m)

#         return _ext.three_interpolate(features, idx, weight)

#     @staticmethod
#     def backward(ctx, grad_out):
#         # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
#         r"""
#         Parameters
#         ----------
#         grad_out : torch.Tensor
#             (B, c, n) tensor with gradients of ouputs

#         Returns
#         -------
#         grad_features : torch.Tensor
#             (B, c, m) tensor with gradients of features

#         None

#         None
#         """
#         idx, weight, m = ctx.three_interpolate_for_backward

#         grad_features = _ext.three_interpolate_grad(
#             grad_out.contiguous(), idx, weight, m
#         )

#         return grad_features, None, None


# three_interpolate = ThreeInterpolate.apply



def grouping_operation(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, nsample]
    Return:
        new_points:, indexed points data, [B, S, C] or [B, S, nsample, C]
    """
    shape = idx.shape
    batch_indices = generate_tensor_batch_indices(shape[0])
    if len(shape) == 2:
        batch_indices = batch_indices.view(shape[0], 1)
    else:
        batch_indices = batch_indices.view(shape[0], 1, 1)
    batch_indices = batch_indices.expand_as(idx)
    index = P.Concat(-1)((batch_indices.reshape(idx.shape + (1,)), idx.reshape(idx.shape + (1,)))) #[B, S, nsample, 2] 2-> (batch_index, point_index)
    new_points = P.GatherNd()(points, index)
    return new_points 




def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * P.BatchMatMul()(src, P.Transpose()(dst, (0, 2, 1)))
    dist += P.Reshape()(P.ReduceSum()(src ** 2, -1), (B, N, 1))
    dist += P.Reshape()(P.ReduceSum()(dst ** 2, -1), (B, 1, M))
    return dist

def ball_query(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = mnp.arange(0, N, 1, ms.int32).view(1, 1, N)
    group_idx = P.Tile()(group_idx, (B, S, 1)) # [B, S, N]
    sqrdists = square_distance(new_xyz, xyz) # [B, S, N]

    idx = sqrdists > radius ** 2
    group_idx = P.Select()(idx, -1 * P.OnesLike()(group_idx), group_idx) # [B, S, N]
    group_idx = P.Cast()(group_idx, ms.float32)
    group_idx, _ = P.TopK()(group_idx, nsample) # [B, S, nsample]
    group_idx = P.Cast()(group_idx, ms.int32)

    group_first = group_idx[:, :, 0].view(B, S, 1)
    group_first = P.Tile()(group_first, (1, 1, nsample))  # [B, S, nsample]

    index = group_idx != -1
    group_first = P.Select()(index, -1 * P.OnesLike()(group_first), group_first)
    group_idx = P.Maximum()(group_idx, group_first) # if there no nsample point within the radius, use multiple first element to replace

    return group_idx


class QueryAndGroup(nn.Cell):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def construct(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, ms.Tensor, ms.Tensor, ms.Tensor) -> Tuple[ms.Tensor]
        r"""
        Parameters
        ----------
        xyz : ms.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : ms.Tensor
            centriods (B, S, 3)
        features : ms.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : ms.Tensor
            (B, 3 + C, S, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz) # [B, S, nsample] use the index 

        # cound and record the unique sample within the radius
        if self.sample_uniformly:
            unique_cnt = mnp.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = P.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = P.randint(0, num_unique, (self.nsample - num_unique,), dtype=ms.int64)
                    all_ind = P.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        grouped_xyz = grouping_operation(xyz, idx)  # (B, S, nsample, 3) real data replace index

        # transpose to (B, 3, S, nsample)
        grouped_xyz = grouped_xyz.swapaxes(2, 3).swapaxes(1, 2)

        # normalize the xyz
        grouped_xyz -= P.swapaxes(new_xyz, 1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            # (B, C, N) -> (B, N, C)
            features = P.swapaxes(features, 1, 2)
            grouped_features = grouping_operation(features, idx) # (B, S, nsample, C) real data replace index
            # transpose to (B, C, S, nsample)
            grouped_features = grouped_features.swapaxes(2, 3).swapaxes(1, 2)
            if self.use_xyz:
                new_features = P.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Cell):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def construct(self, xyz, new_xyz, features=None):
        # type: (GroupAll, ms.Tensor, ms.Tensor, ms.Tensor) -> Tuple[ms.Tensor]
        r"""
        Parameters
        ----------
        xyz : ms.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : ms.Tensor
            Ignored
        features : ms.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : ms.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.swapaxes(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = P.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features


# class CylinderQuery(Function):
#     @staticmethod
#     def forward(ctx, radius, hmin, hmax, nsample, xyz, new_xyz, rot):
#         # type: (Any, float, float, float, int, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
#         r"""

#         Parameters
#         ----------
#         radius : float
#             radius of the cylinders
#         hmin, hmax : float
#             endpoints of cylinder height in x-rotation axis
#         nsample : int
#             maximum number of features in the cylinders
#         xyz : torch.Tensor
#             (B, N, 3) xyz coordinates of the features
#         new_xyz : torch.Tensor
#             (B, npoint, 3) centers of the cylinder query
#         rot: torch.Tensor
#             (B, npoint, 9) flatten rotation matrices from
#                            cylinder frame to world frame

#         Returns
#         -------
#         torch.Tensor
#             (B, npoint, nsample) tensor with the indicies of the features that form the query balls
#         """
#         return _ext.cylinder_query(new_xyz, xyz, rot, radius, hmin, hmax, nsample)

#     @staticmethod
#     def backward(ctx, a=None):
#         return None, None, None, None, None, None, None


# cylinder_query = CylinderQuery.apply


# class CylinderQueryAndGroup(nn.Module):
#     r"""
#     Groups with a cylinder query of radius and height

#     Parameters
#     ---------
#     radius : float32
#         Radius of cylinder
#     hmin, hmax: float32
#         endpoints of cylinder height in x-rotation axis
#     nsample : int32
#         Maximum number of features to gather in the ball
#     """

#     def __init__(self, radius, hmin, hmax, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, rotate_xyz=True, sample_uniformly=False, ret_unique_cnt=False):
#         # type: (CylinderQueryAndGroup, float, float, float, int, bool) -> None
#         super(CylinderQueryAndGroup, self).__init__()
#         self.radius, self.nsample, self.hmin, self.hmax, = radius, nsample, hmin, hmax
#         self.use_xyz = use_xyz
#         self.ret_grouped_xyz = ret_grouped_xyz
#         self.normalize_xyz = normalize_xyz
#         self.rotate_xyz = rotate_xyz
#         self.sample_uniformly = sample_uniformly
#         self.ret_unique_cnt = ret_unique_cnt
#         if self.ret_unique_cnt:
#             assert(self.sample_uniformly)

#     def forward(self, xyz, new_xyz, rot, features=None):
#         # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
#         r"""
#         Parameters
#         ----------
#         xyz : torch.Tensor
#             xyz coordinates of the features (B, N, 3)
#         new_xyz : torch.Tensor
#             centriods (B, npoint, 3)
#         rot : torch.Tensor
#             rotation matrices (B, npoint, 3, 3)
#         features : torch.Tensor
#             Descriptors of the features (B, C, N)

#         Returns
#         -------
#         new_features : torch.Tensor
#             (B, 3 + C, npoint, nsample) tensor
#         """
#         B, npoint, _ = new_xyz.size()
#         idx = cylinder_query(self.radius, self.hmin, self.hmax, self.nsample, xyz, new_xyz, rot.view(B, npoint, 9))

#         if self.sample_uniformly:
#             unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
#             for i_batch in range(idx.shape[0]):
#                 for i_region in range(idx.shape[1]):
#                     unique_ind = torch.unique(idx[i_batch, i_region, :])
#                     num_unique = unique_ind.shape[0]
#                     unique_cnt[i_batch, i_region] = num_unique
#                     sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
#                     all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
#                     idx[i_batch, i_region, :] = all_ind


#         xyz_trans = xyz.transpose(1, 2).contiguous()
#         grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
#         grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
#         if self.normalize_xyz:
#             grouped_xyz /= self.radius
#         if self.rotate_xyz:
#             grouped_xyz_ = grouped_xyz.permute(0, 2, 3, 1).contiguous() # (B, npoint, nsample, 3)
#             grouped_xyz_ = torch.matmul(grouped_xyz_, rot)
#             grouped_xyz = grouped_xyz_.permute(0, 3, 1, 2).contiguous()


#         if features is not None:
#             grouped_features = grouping_operation(features, idx)
#             if self.use_xyz:
#                 new_features = torch.cat(
#                     [grouped_xyz, grouped_features], dim=1
#                 )  # (B, C + 3, npoint, nsample)
#             else:
#                 new_features = grouped_features
#         else:
#             assert (
#                 self.use_xyz
#             ), "Cannot have not features and not use xyz as a feature!"
#             new_features = grouped_xyz

#         ret = [new_features]
#         if self.ret_grouped_xyz:
#             ret.append(grouped_xyz)
#         if self.ret_unique_cnt:
#             ret.append(unique_cnt)
#         if len(ret) == 1:
#             return ret[0]
#         else:
#             return tuple(ret)