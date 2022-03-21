"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import cv2
import sys
import copy
import os
import os.path as osp
from plyfile import PlyData, PlyElement
import json
import glob
import tqdm

from util import *
# from mesh_filter import *

import trimesh

ROOT_FOLDER =  '/data/ScanNet/ScanNet_raw_data/scannet/scans/' #'/data/ScanNet/ScanNet_raw_data/scannet/' #"/data/ScanNet/Atlas/data_preprocess_debug/scannet/"
DUMP_FOLDER = '/data/ScanNet/ScanNet_raw_data/video_plane_fitting/'
#numPlanes = 200
numPlanesPerSegment = 2
planeAreaThreshold = 10
MergedPlaneAreaThreshold = 120
numIterations = 1000
numIterationsPair = 1000
planeDiffThreshold = 0.05
fittingErrorThreshold = planeDiffThreshold

orthogonalThreshold = np.cos(np.deg2rad(60))
parallelThreshold = np.cos(np.deg2rad(30))

debug = False
debugIndex = -1


def loadClassMap():
    classMap = {}
    classLabelMap = {}
    with open(ROOT_FOLDER + 'scannetv2-labels.combined.tsv') as info_file:
        line_index = 0
        for line in info_file:
            if line_index > 0:
                line = line.split('\t')

                key = line[1].strip()
                classMap[key] = line[7].strip()
                classMap[key + 's'] = line[7].strip()
                classMap[key + 'es'] = line[7].strip()
                classMap[key[:-1] + 'ves'] = line[7].strip()

                if line[4].strip() != '':
                    nyuLabel = int(line[4].strip())
                else:
                    nyuLabel = -1
                    pass
                classLabelMap[key] = [nyuLabel, line_index - 1]
                classLabelMap[key + 's'] = [nyuLabel, line_index - 1]
                classLabelMap[key[:-1] + 'ves'] = [nyuLabel, line_index - 1]
                pass
            line_index += 1

        classMap['unannotated'] = 'unannotated'
        classMap['board'] = 'whiteboard'
        classMap['blackboard'] = 'whiteboard'
        classMap['bulletin board'] = 'whiteboard'
        classMap['storage bin'] = 'cabinet'

    return classMap, classLabelMap


def mergePlanes(points, normals, planes, planePointIndices, planeSegments, segmentNeighbors, numPlanes, angleThres = np.cos(np.deg2rad(30)), debug=False):

    planeFittingErrors = []
    for plane, pointIndices in zip(planes, planePointIndices):
        XYZ = points[pointIndices]
        planeNorm = np.linalg.norm(plane)
        if planeNorm == 0:
            planeFittingErrors.append(fittingErrorThreshold * 3)
            continue
        diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / planeNorm
        planeFittingErrors.append(diff.mean())


    planeList = zip(planes, planePointIndices, planeSegments, planeFittingErrors)
    planeList = sorted(planeList, key=lambda x:x[3]) # the less the fitting error, the topper it is

    while len(planeList) > 0:
        hasChange = False
        planeIndex = 0

        if debug:
            for index, planeInfo in enumerate(sorted(planeList, key=lambda x:-len(x[1]))):
                print(index, planeInfo[0] / np.linalg.norm(planeInfo[0]), planeInfo[2], planeInfo[3])
                continue
            pass

        while planeIndex < len(planeList):
            plane, pointIndices, segments, fittingError = planeList[planeIndex]
            if fittingError > fittingErrorThreshold:
                break
            neighborSegments = []
            for segment in segments:
                if segment in segmentNeighbors:
                    neighborSegments += segmentNeighbors[segment]

            neighborSegments += list(segments)
            neighborSegments = set(neighborSegments)

            # for each candidate plane, only merge with its best neighbor
            bestNeighborPlane = (fittingErrorThreshold, -1, None)
            for neighborPlaneIndex, neighborPlane in enumerate(planeList):
                # only merge with the planes whose fitting error larger than itself
                if neighborPlaneIndex <= planeIndex:
                    continue

                # if the two planes do not share the same segment, skip merging
                if not bool(neighborSegments & neighborPlane[2]):
                    continue

                # the neighbor plane should be valid
                neighborPlaneNorm = np.linalg.norm(neighborPlane[0])
                if neighborPlaneNorm < 1e-4:
                    continue

                dotProduct = np.abs(np.dot(neighborPlane[0], plane) / np.maximum(neighborPlaneNorm * np.linalg.norm(plane), 1e-4))

                # if the angle is too large, skip merging the two planes
                if dotProduct < orthogonalThreshold:
                    continue

                newPointIndices = np.concatenate([neighborPlane[1], pointIndices], axis=0)
                XYZ = points[newPointIndices]
                norm = normals[newPointIndices]
                # only fit a new plane when neighbouring plane is large and close enough (maybe could improve)
                #if dotProduct > parallelThreshold and len(neighborPlane[1]) > len(pointIndices) * 0.5:
                if dotProduct > angleThres:
                    newPlane = fitPlane(XYZ, norm)  #todo: use ransac
                else:
                    continue

                diff = np.abs(np.matmul(XYZ, newPlane) - np.ones(XYZ.shape[0])) / np.linalg.norm(newPlane)
                newFittingError = diff.mean()
                if debug:
                    print(len(planeList), planeIndex, neighborPlaneIndex, newFittingError, plane / np.linalg.norm(plane), neighborPlane[0] / np.linalg.norm(neighborPlane[0]), dotProduct, orthogonalThreshold)
                    pass
                if newFittingError < bestNeighborPlane[0]:
                    newPlaneInfo = [newPlane, newPointIndices, segments.union(neighborPlane[2]), newFittingError]
                    bestNeighborPlane = (newFittingError, neighborPlaneIndex, newPlaneInfo)

            if bestNeighborPlane[1] != -1:
                newPlaneList = planeList[:planeIndex] + planeList[planeIndex + 1:bestNeighborPlane[1]] + planeList[bestNeighborPlane[1] + 1:]

                # here newplane becomes newplaneinfo, not only parameters
                newFittingError, newPlaneIndex, newPlane = bestNeighborPlane
                for newPlaneIndex in range(len(newPlaneList)):
                    if (newPlaneIndex == 0 and newPlaneList[newPlaneIndex][3] > newFittingError) \
                       or newPlaneIndex == len(newPlaneList) - 1 \
                       or (newPlaneList[newPlaneIndex][3] < newFittingError and newPlaneList[newPlaneIndex + 1][3] > newFittingError):
                        newPlaneList.insert(newPlaneIndex, newPlane)
                        break
                    continue
                if len(newPlaneList) == 0:
                    newPlaneList = [newPlane]

                planeList = newPlaneList
                hasChange = True
            else:
                planeIndex += 1

            continue

        if not hasChange:
            break
        continue

    planeList = sorted(planeList, key=lambda x:-len(x[1]))


    minNumPlanes, maxNumPlanes = numPlanes
    if minNumPlanes == 1 and len(planeList) == 0:
        if debug:
            print('at least one plane')
            pass
    elif len(planeList) > maxNumPlanes:
        if debug:
            print('too many planes', len(planeList), maxNumPlanes)
            pass
        planeList = planeList[:maxNumPlanes] + [(np.zeros(3), planeInfo[1], planeInfo[2], fittingErrorThreshold) for planeInfo in planeList[maxNumPlanes:]]


    groupedPlanes, groupedPlanePointIndices, groupedPlaneSegments, groupedPlaneFittingErrors = zip(*planeList)
    return groupedPlanes, groupedPlanePointIndices, groupedPlaneSegments


def get_meshNorm(points, plydata):
    mesh = trimesh.Trimesh(
        vertices=points, faces=np.array([f for f in plydata['face'].data['vertex_indices']]), process=False)
    tmp_normal = mesh.vertex_normals # this is read_only

    # vis_meshNormal(mesh, tmp_normal)
    # vis_meshNormal(mesh, np.abs(tmp_normal))
    # del mesh
    return mesh, tmp_normal

def process_mesh(scene_id):
    if os.path.isfile(DUMP_FOLDER + '/mesh/{}_planes.ply'.format(scene_id)):
        return
    # ==============================
    # load instance label agregated idx
    # offer the instance id, and all the over_segmt_pieceID belong to this instance
    # ==============================
    filename = ROOT_FOLDER + scene_id + '/' + scene_id + '.aggregation.json'
    data = json.load(open(filename, 'r'))
    aggregation = np.array(data['segGroups'])

    # the object/instance id is in ascending order for sure
    groupSegments = []      # the over_sgmts_id in the instance, groupSegments
    groupLabels = []        # the instance semantic label, groupLabels
    # instance-level
    for segmentIndex in range(len(aggregation)):
        groupSegments.append(aggregation[segmentIndex]['segments'])
        groupLabels.append(aggregation[segmentIndex]['label'])

    # ==============================
    # load ply
    # ==============================
    filename = os.path.join(ROOT_FOLDER, scene_id, scene_id + '_vh_clean_2.ply') #labels.


    plydata = PlyData.read(filename)
    vertices = plydata['vertex']
    points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
    faces = np.array(plydata['face']['vertex_indices'])

    mesh, normals = get_meshNorm(points, plydata)# 2 parallel but opposite plane will have same normal, it is ok in the fitting case
    # mesh.visual.vertex_colors = 1
    # mesh.show()
    # ==============================
    # load over_segment information
    # the number of segmentation is same as vertice number, each is a over_segmt_id to the vertex
    # ==============================
    filename = os.path.join(ROOT_FOLDER, scene_id, scene_id + '_vh_clean_2.0.010000.segs.json')
    data = json.load(open(filename, 'r'))
    segmentation = np.array(data['segIndices']).astype(np.int32)  # the pnt --> over-segmt mapping, segmentation

    # get the segmt number and unannotated segmts (saved in uniqueSegments)
    uniqueSegments = np.unique(segmentation).tolist()
    numSegments = 0
    for segments in groupSegments:
        for segmentIndex in segments:
            if segmentIndex in uniqueSegments:
                uniqueSegments.remove(segmentIndex)

        numSegments += len(segments)


    # the rest are unannotated segments, not belong to any instances, record the over-segmts has not semantic label
    for segment in uniqueSegments:
        groupSegments.append([segment, ]) #  # for unannotated sgmts, each sgmt is treated as an individual object
        groupLabels.append('unannotated')
        continue


    # nyu labels
    class_map, class_label_map = loadClassMap() #class_label_map never used later
    nyu_labels = []
    for origin_label in groupLabels:
        nyu_label = class_map[origin_label]
        nyu_labels.append(nyu_label)

    #map nyu_labels
    plane_label_map = load_plane_map()
    num_map = load_num_map()

    label_idx_map = {}
    num_map.pop('non_plane')
    for idx, label in enumerate(num_map):
        label_idx_map[label] = idx + 1

    #label_idx_map = {label:idx+1 for idx,label in enumerate(num_map)}
    label_idx_map['non_plane'] = 0

    transformed_groupLabels = []

    for label in nyu_labels:
        transformed_groupLabels.append(plane_label_map[label])


    segmentEdges = []
    for faceIndex in range(faces.shape[0]):
        # the three elements of each face are their point indices
        face = faces[faceIndex]
        segment_1 = segmentation[face[0]]
        segment_2 = segmentation[face[1]]
        segment_3 = segmentation[face[2]]
        if segment_1 != segment_2 or segment_1 != segment_3:
            if segment_1 != segment_2 and segment_1 != -1 and segment_2 != -1:
                segmentEdges.append((min(segment_1, segment_2), max(segment_1, segment_2)))

            if segment_1 != segment_3 and segment_1 != -1 and segment_3 != -1:
                segmentEdges.append((min(segment_1, segment_3), max(segment_1, segment_3)))

            if segment_2 != segment_3 and segment_2 != -1 and segment_3 != -1:
                segmentEdges.append((min(segment_2, segment_3), max(segment_2, segment_3)))


    # neighbouring segments
    segmentEdges = list(set(segmentEdges))

    segmentNeighbors = {}
    for segmentEdge in segmentEdges:
        if segmentEdge[0] not in segmentNeighbors:
            segmentNeighbors[segmentEdge[0]] = []

        segmentNeighbors[segmentEdge[0]].append(segmentEdge[1])

        if segmentEdge[1] not in segmentNeighbors:
            segmentNeighbors[segmentEdge[1]] = []

        segmentNeighbors[segmentEdge[1]].append(segmentEdge[0])


    planeGroups = []
    print('num groups', len(groupSegments)) # the over_sgmts_id in the instance, one row represent one instance

    labelNumPlanes = load_num_map()
    allXYZ = points.reshape(-1,3)


    for groupIndex, group in enumerate(groupSegments):
        groupLabel = transformed_groupLabels[groupIndex]
        groupLabelIndex = label_idx_map[groupLabel]
        minNumPlanes, maxNumPlanes = labelNumPlanes[groupLabel]

        # adjust the error scale w.r.t. label
        fittingErrorScale = 1
        angleErrorThres = parallelThreshold

        if groupLabel == 'floor':
            ## Relax the constraint for the floor due to the misalignment issue in ScanNet
            fittingErrorScale = 10
        elif groupLabel == 'wall':
            fittingErrorScale = 1.8
            angleErrorThres =  np.cos(np.deg2rad(20))
        elif groupLabel == 'cabinet':
            fittingErrorScale = 1.5
            angleErrorThres = np.cos(np.deg2rad(25))



        # for non-plane object, directly skip
        if maxNumPlanes == 0:
            pointMasks = []
            for segmentIndex in group:
                pointMasks.append(segmentation == segmentIndex)

            pointIndices = np.any(np.stack(pointMasks, 0), 0).nonzero()[0]
            # plane params, 3D point indices and segments and semantic label idx
            groupPlanes = [[np.zeros(3), pointIndices, [], groupLabelIndex, []]]
            planeGroups.append(groupPlanes)

        # for object may have plane
        groupPlanes = []                    # all planes in current obj
        groupPlanePointIndices = []         # pnt_idx corresponding to each plane
        groupPlaneSegments = []             # sgmt_idx corresponding to each plane



        #  for each segmt, fit one plane first, and then merge them
        for segmentIndex in group:
            # segmentMask select all vertex belong to current over-segmt
            segmentMask = segmentation == segmentIndex # segmentation is the per_vertex over-sgment id,
            allSegmentIndices = segmentMask.nonzero()[0] #return idx of non zero items
            segmentIndices = allSegmentIndices.copy()

            XYZ = allXYZ[segmentMask]
            norm = normals[segmentMask]
            numPoints = XYZ.shape[0]

            for c in range(2): #There is maximum 2 for each over_segmt
                if c == 0:
                    ## First try to fit one plane
                    # print('run svd')
                    plane = fitPlane(XYZ, norm)
                    diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)
                    # while theortically, we should compute the np.abs(norm @ plane_norm), we emprically find do abs to
                    # both variables first lead to better result
                    if diff.mean() < fittingErrorThreshold * fittingErrorScale and \
                            np.sum((np.abs(norm).dot(np.abs(plane/np.linalg.norm(plane)).T)) > angleErrorThres) > 0.9* (XYZ.shape[0])and \
                            len(XYZ) >= planeAreaThreshold:

                        groupPlanes.append(plane)
                        groupPlanePointIndices.append(segmentIndices)
                        groupPlaneSegments.append(set([segmentIndex]))
                        # print('svd accepted!')

                        break
                else:
                    ## Run ransac
                    # print('run_ransac')
                    # segmentPlanes, segmentPlanePointIndices, remain_sgmt_pntsId = planeRCNN_ransac(XYZ, segmentIndices,
                    #                                                                         n_iter =numIterations, norm_in=norm, n_plane=2,
                    #                                                                         diff_thres=planeDiffThreshold,
                    #                                                                         area_thres=planeAreaThreshold,
                    #                                                                         norm_thres=parallelThreshold)


                    segmentPlanes, segmentPlanePointIndices, remain_sgmt_pntsId = onePoint_ransac(XYZ, segmentIndices,
                                                                                                   n_iter = numIterations,
                                                                                                   norm_in=norm, n_plane=2,
                                                                                                   diff_thres=fittingErrorThreshold*fittingErrorScale,
                                                                                                   area_thres=planeAreaThreshold,
                                                                                                  norm_thres=angleErrorThres)


                    # print(segmentPlanes, segmentPlanes2)
                    # if only less than half points of this segment are fitted, record an invalid plane.
                    if sum([len(indices) for indices in segmentPlanePointIndices]) < numPoints * 0.5:
                        groupPlanes.append(np.zeros(3))
                        groupPlanePointIndices.append(allSegmentIndices)
                        groupPlaneSegments.append(set([segmentIndex]))

                    else:
                        if len(remain_sgmt_pntsId) > 0: #shit I used segmentIndices before
                            ## Add remaining non-planar regions
                            segmentPlanes.append(np.zeros(3))
                            segmentPlanePointIndices.append(remain_sgmt_pntsId)

                        groupPlanes += segmentPlanes                            # the ransac result n/d
                        groupPlanePointIndices += segmentPlanePointIndices       # the points belong to which segment idx

                        for _ in range(len(segmentPlanes)):
                            groupPlaneSegments.append(set([segmentIndex]))
                            continue


        # groupPlanePointIndices: all the point indices belong to this instance
        numRealPlanes = len([plane for plane in groupPlanes if np.linalg.norm(plane) > 1e-4])
        if minNumPlanes == 1 and numRealPlanes == 0:
            ## Some instances always contain at least one planes (e.g, the floor)
            # if it fail to get a qualified plane, use the large plane area in current record to fit one again
            maxArea = (planeAreaThreshold, -1)
            for index, indices in enumerate(groupPlanePointIndices):
                if len(indices) > maxArea[0]:
                    maxArea = (len(indices), index)

            maxArea, planeIndex = maxArea
            # keep the plane area threshold valid
            if planeIndex >= 0:
                groupPlanes[planeIndex] = fitPlane(allXYZ[groupPlanePointIndices[planeIndex]])
                numRealPlanes = 1

        if minNumPlanes == 1 and maxNumPlanes == 1 and numRealPlanes > 1:
            ## Some instances always contain at most one planes (e.g, the floor)

            # just fit one plane with all fitted points
            pointIndices = np.concatenate([indices for plane, indices in zip(groupPlanes, groupPlanePointIndices)], axis=0)
            XYZ = allXYZ[pointIndices]
            plane = fitPlane(XYZ, normals[pointIndices])
            diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)


            # try to fit one plane with all fitted points, if the fitted plane satisfies error threshold
            if diff.mean() < fittingErrorThreshold * fittingErrorScale:
                groupPlanes = [plane]
                groupPlanePointIndices = [pointIndices]
                planeSegments = []
                for segments in groupPlaneSegments:
                    planeSegments += list(segments)
                    continue
                groupPlaneSegments = [set(planeSegments)]
                numRealPlanes = 1

        # for debug
        # if groupLabel == 'cabinet':
        #     colorMap = ColorPalette(len(groupPlanePointIndices))
        #     mesh.visual.vertex_colors = [102, 102, 102, 255]
        #     for i, pntId in enumerate(groupPlanePointIndices):
        #         print(groupPlanes[i] / np.linalg.norm(groupPlanes[i]), pntId.shape[0])
        #         mesh.visual.vertex_colors[pntId] = np.concatenate([colorMap.colorMap[i], np.array([255])])
        #         # if pntId.shape[0] > 1000:
        #     mesh.show()

        # consider merge different planes belong to the same instance together
        if numRealPlanes > 1:
            groupPlanes, groupPlanePointIndices, groupPlaneSegments = mergePlanes(points, normals, groupPlanes,
                                                                                  groupPlanePointIndices, groupPlaneSegments,
                                                                                 segmentNeighbors, angleThres=angleErrorThres,
                                                                                  numPlanes=(minNumPlanes, maxNumPlanes), debug=debugIndex != -1)
        # if groupLabel == 'cabinet':
        #     colorMap = ColorPalette(len(groupPlanePointIndices))
        #     mesh.visual.vertex_colors = [102, 102, 102, 255]
        #     for i, pntId in enumerate(groupPlanePointIndices):
        #         print(groupPlanes[i]/np.linalg.norm(groupPlanes[i]), pntId.shape[0])
        #         mesh.visual.vertex_colors[pntId] = np.concatenate([colorMap.colorMap[i], np.array([255])])
        #         # if pntId.shape[0] > 1000:
        #     mesh.show()

        # get neighbor planes idx
        groupNeighbors = []
        for planeIndex, planeSegments in enumerate(groupPlaneSegments):
            neighborSegments = []
            for segment in planeSegments:
                if segment in segmentNeighbors:
                    neighborSegments += segmentNeighbors[segment]
                    pass
                continue
            neighborSegments += list(planeSegments)
            neighborSegments = set(neighborSegments)
            neighborPlaneIndices = []
            for neighborPlaneIndex, neighborPlaneSegments in enumerate(groupPlaneSegments):
                if neighborPlaneIndex == planeIndex:
                    continue

                # fitted plane and presaved neighbor segments have intersection
                if bool(neighborSegments & neighborPlaneSegments):
                    plane = groupPlanes[planeIndex]
                    neighborPlane = groupPlanes[neighborPlaneIndex]
                    if np.linalg.norm(plane) * np.linalg.norm(neighborPlane) < 1e-4:
                        continue
                    dotProduct = np.abs(np.dot(plane, neighborPlane) / np.maximum(np.linalg.norm(plane) * np.linalg.norm(neighborPlane), 1e-4))
                    if dotProduct < orthogonalThreshold:
                        neighborPlaneIndices.append(neighborPlaneIndex)

            groupNeighbors.append(neighborPlaneIndices)


        groupPlaneLabels = [groupLabelIndex for _ in range(len(groupPlanes))]

        groupPlaneInfos = zip(groupPlanes, groupPlanePointIndices, groupNeighbors, groupPlaneLabels, groupPlaneSegments)
        planeGroups.append(groupPlaneInfos)


    planeGroups_copy = copy.deepcopy(planeGroups)
    numPlanes = sum([len(list(group)) for group in planeGroups_copy])

    # the first '+1' makes a space for the white color(non-plane), the second '+1' makes the first plane different from background
    # plane_id: segmentationColor // 100 - 1
    segmentationColor = (np.arange(numPlanes + 1) + 1) * 100
    colorMap = np.stack([segmentationColor / (256 * 256), segmentationColor / 256 % 256, segmentationColor % 256], axis=1)
    colorMap[-1] = 255


    planes = []
    planePointIndices = []
    planeLabels = []
    planeSegments = []
    planeInfo = []
    structureIndex = 0

    # instance-level
    for index, group in enumerate(planeGroups):
        groupPlanes, groupPlanePointIndices, groupNeighbors, groupPlaneLabels, groupPlaneSegments = zip(*group)

        diag = np.diag(np.ones(len(groupNeighbors)))
        adjacencyMatrix = diag.copy()
        for groupIndex, neighbors in enumerate(groupNeighbors):
            for neighbor in neighbors:
                adjacencyMatrix[groupIndex][neighbor] = 1 # for plane id, record this neighbor idx


        groupPlaneIndices = (adjacencyMatrix.sum(-1) >= 2).nonzero()[0]
        planes += groupPlanes
        planePointIndices += groupPlanePointIndices
        planeSegments += groupPlaneSegments
        planeLabels += groupPlaneLabels
        #planeInfo += groupInfo


    planar_points_num = 0

    # non-plane pixels(-1) will be rendered as (255,255,255),
    planeSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
    planeidx_to_semanticidx = dict()
    main_plane_indexs = []
    for planeIndex, (planePoints, planeLabel) in enumerate(zip(planePointIndices, planeLabels)):
        # invalid plane param will be skipped,
        if np.linalg.norm(planes[planeIndex]) < 1e-10 or planePoints.shape[0] < MergedPlaneAreaThreshold:
            continue

        else:
            planeSegmentation[planePoints] = planeIndex # plane saving list
            planeidx_to_semanticidx[planeIndex] = planeLabel

            if planeLabel != 14:
                main_plane_indexs.append(planeIndex)
                #planar_points_num += len(planePoint)

    for planeIndex in np.unique(planeSegmentation):
        if planeIndex in main_plane_indexs:
            planar_points_num += np.sum(planeSegmentation == planeIndex)

    with open('planar_points.txt', 'a') as fp:
        fp.write(scene_id)
        fp.write('\t')
        fp.write(str(planar_points_num))
        fp.write('\t')
        fp.write(str(points.shape[0]))
        fp.write('\n')



    planes = np.array(planes)
    print('number of planes: ', planes.shape[0])
    planesD = 1.0 / np.maximum(np.linalg.norm(planes, axis=-1, keepdims=True), 1e-4) # d
    planes *= pow(planesD, 2) #nd

    print('valid number of planes:', len(planeidx_to_semanticidx.keys()))

    # remove faces whose 3 vertices have different label
    removeIndices = []
    for faceIndex in range(faces.shape[0]):
        face = faces[faceIndex]
        segment_1 = planeSegmentation[face[0]]
        segment_2 = planeSegmentation[face[1]]
        segment_3 = planeSegmentation[face[2]]
        if segment_1 != segment_2 or segment_1 != segment_3:
            removeIndices.append(faceIndex)

    faces = np.delete(faces, removeIndices)

    #todo clean up the flying pieces

    colors = colorMap[planeSegmentation]
    # writePointCloudFace(annotationFolder + '/planes.ply', np.concatenate([points, colors], axis=-1), faces)
    if not os.path.isdir(DUMP_FOLDER + '/mesh'):
        os.makedirs(DUMP_FOLDER + '/mesh')
    writePointCloudFace(DUMP_FOLDER + '/mesh/{}_planes.ply'.format(scene_id), np.concatenate([points, colors], axis=-1), faces)


    if not os.path.isdir(DUMP_FOLDER + '/plane_param'):
        os.makedirs(DUMP_FOLDER + '/plane_param')
    np.save(DUMP_FOLDER + '/plane_param/{}_planes.npy'.format(scene_id), planes) # save as n*d
    #np.save(annotationFolder + '/plane_info.npy', planeInfo)

    # if not os.path.isdir(DUMP_FOLDER + '/plane_semseg_map'):
    #     os.makedirs(DUMP_FOLDER + '/plane_semseg_map')
    # with open(DUMP_FOLDER + '/plane_semseg_map/{}_semantic_mapping.txt'.format(scene_id),'w') as fp:
    #     for key,val in planeidx_to_semanticidx.items():
    #         fp.write(str(key)) #plane ID, semantic id
    #         fp.write('\t')
    #         fp.write(str(val))
    #         fp.write('\n')

    return


if __name__=='__main__':

    train_scenes =  [x for x in os.listdir(ROOT_FOLDER) if 'scene' in x and 'tar' not in x]#, '' , 'scene0002_00', 'scene0003_00'
    train_scenes = sorted(train_scenes)#[:1]
    for scene_id in tqdm.tqdm(train_scenes):
        # print(scene_id)
        # segmentation_folder = osp.join(ROOT_FOLDER + '/' + scene_id + '/new_annotation/segmentation')
        # if osp.exists(segmentation_folder):
        #     cmd = 'rm -r ' + segmentation_folder
        #     os.system(cmd)

        process_mesh(scene_id)

        #segmentation_folder = osp.join(ROOT_FOLDER + '/' + scene_id + '/new_annotation/segmentation')
        #if not osp.exists(segmentation_folder):
        #    os.makedirs(segmentation_folder)

        # cmd = './Renderer/Renderer --scene_id=' + scene_id + ' --root_folder=' + ROOT_FOLDER + ' --frame_stride 5'
        #ret = os.system(cmd)

        #if ret != 0:
        #    break

