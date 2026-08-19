"""ARDY Core -> Blender FK bridge utilities."""
from __future__ import annotations
import numpy as np
import bpy, mathutils

C=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
MIXAMO_COMMON={
"Hips","Spine","Spine1","Spine2","Neck","Head",
"RightShoulder","RightArm","RightForeArm","RightHand","RightHandThumb1",
"LeftShoulder","LeftArm","LeftForeArm","LeftHand","LeftHandThumb1",
"RightUpLeg","RightLeg","RightFoot","RightToeBase",
"LeftUpLeg","LeftLeg","LeftFoot","LeftToeBase"}
PREFERRED_CHILD={
"Hips":"Spine","Spine":"Spine1","Spine1":"Spine2","Spine2":"Spine3","Spine3":"Neck","Neck":"Head",
"RightShoulder":"RightArm","RightArm":"RightForeArm","RightForeArm":"RightHand","RightHand":"RightHandEnd",
"LeftShoulder":"LeftArm","LeftArm":"LeftForeArm","LeftForeArm":"LeftHand","LeftHand":"LeftHandEnd",
"RightUpLeg":"RightLeg","RightLeg":"RightFoot","RightFoot":"RightToeBase",
"LeftUpLeg":"LeftLeg","LeftLeg":"LeftFoot","LeftFoot":"LeftToeBase"}

def cvt_pos(v,scale): return (C@np.asarray(v,dtype=np.float64))*scale
def cvt_rot(r):
    r=np.asarray(r,dtype=np.float64); return C@r@C.T
def source_bone_name(name): return f"mixamorig:{name}" if name in MIXAMO_COMMON else f"ARDY_{name}"

def build_children(parents):
    children=[[] for _ in parents]
    for i,p in enumerate(parents):
        if int(p)>=0: children[int(p)].append(i)
    return children

def choose_tail(i,names,parents,children,points,scale):
    idx={n:k for k,n in enumerate(names)}
    preferred=PREFERRED_CHILD.get(names[i])
    if preferred in idx and idx[preferred] in children[i]: return points[idx[preferred]].copy()
    if children[i]: return points[children[i][0]].copy()
    p=int(parents[i])
    if p>=0:
        d=points[i]-points[p]; n=np.linalg.norm(d)
        if n>1e-8: return points[i]+d/n*(.10*scale)
    return points[i]+np.array([0.,.10*scale,0.])

def validate_npz(data):
    required={"joint_names","joint_parents","neutral_joints","local_rot_mats","root_positions","posed_joints","fps"}
    missing=required-data.keys()
    if missing: raise KeyError(f"Missing bridge keys: {sorted(missing)}")
    names=[str(x) for x in data["joint_names"].tolist()]; j=len(names)
    local=np.asarray(data["local_rot_mats"]); t=local.shape[0] if local.ndim else 0
    expected={"local_rot_mats":(t,j,3,3),"root_positions":(t,3),"posed_joints":(t,j,3),
              "neutral_joints":(j,3),"joint_parents":(j,)}
    for k,shape in expected.items():
        if np.asarray(data[k]).shape!=shape: raise ValueError(f"{k} shape {np.asarray(data[k]).shape}, expected {shape}")
        if k!="joint_parents" and not np.isfinite(np.asarray(data[k])).all(): raise ValueError(f"{k} contains NaN/Inf")
    parents=np.asarray(data["joint_parents"]); fps=float(np.asarray(data["fps"]).reshape(-1)[0])
    if np.sum(parents<0)!=1: raise ValueError("Expected exactly one root joint")
    if not fps>0: raise ValueError(f"Invalid fps: {fps}")
    return t,j,fps

def set_frame(arm,fi,local_rots,roots,rest_rots,names,root_idx,scale,prev,keys):
    for j,name in enumerate(names):
        pb=arm.pose.bones[name]; r0=rest_rots[j]
        q=mathutils.Matrix((r0.T@cvt_rot(local_rots[fi,j])@r0).tolist()).to_quaternion(); q.normalize()
        if prev[j] is not None and q.dot(prev[j])<0: q.negate()
        prev[j]=q.copy(); pb.rotation_mode="QUATERNION"; pb.rotation_quaternion=q
        pb.scale=(1.,1.,1.); pb.location=(0.,0.,0.)
        if j==root_idx: pb.location=tuple(r0.T@cvt_pos(roots[fi],scale))
        if keys:
            pb.keyframe_insert(data_path="rotation_quaternion",frame=fi+1)
            if j==root_idx: pb.keyframe_insert(data_path="location",frame=fi+1)

def validate_fk(scene,arm,samples,local_rots,roots,posed,rest_rots,names,root_idx,scale,tolerance):
    for f in samples:
        set_frame(arm,f,local_rots,roots,rest_rots,names,root_idx,scale,[None]*len(names),False)
        scene.frame_set(f+1); bpy.context.view_layer.update()
        actual=np.stack([np.asarray(arm.pose.bones[n].head) for n in names])
        expected=np.stack([cvt_pos(v,scale) for v in posed[f]])
        errors=np.linalg.norm(actual-expected,axis=1); mx=float(errors.max())
        if mx>tolerance*max(scale,1e-8):
            w=int(np.argmax(errors)); raise RuntimeError(f"FK validation failed frame {f}: {mx:.6f} on {names[w]}")
    print(f"FK bridge validation passed on {len(samples)} frame(s).")
