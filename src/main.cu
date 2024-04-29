// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// Authors: Alok Vermaal, Alok.Verma@cs.tum.edu
//          Julio Oscanoa, julio.oscanoa@tum.de
//          Miguel Trasobares, miguel.trasobares@tum.de
// Supervisors: Robert Maier, robert.maier@in.tum.de
//              Christiane Sommer, sommerc@in.tum.de
// Killing fusion main program
// ########################################################################
#include <iostream>
#include <vector>

#include "mat.h"

#include <cuda_runtime.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "helper.cuh"
#include "dataset.h"
#include "tsdf_volume.h"
#include "marching_cubes.h"
#include "optimizer.cuh"

#include <open3d/Open3D.h>


#define STR1(x)  #x
#define STR(x)  STR1(x)


typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;


bool depthToVertexMap(const Mat3f &K, const cv::Mat &depth, cv::Mat &vertexMap)
{
    if (depth.type() != CV_32FC1 || depth.empty())
        return false;

    int w = depth.cols;
    int h = depth.rows;
    vertexMap = cv::Mat::zeros(h, w, CV_32FC3);
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    float fxInv = 1.0f / fx;
    float fyInv = 1.0f / fy;
    float* ptrVert = (float*)vertexMap.data;

    const float* ptrDepth = (const float*)depth.data;
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            float depthMeter = ptrDepth[y*w + x];
            float x0 = (float(x) - cx) * fxInv;
            float y0 = (float(y) - cy) * fyInv;

            size_t off = (y*w + x) * 3;
            ptrVert[off] = x0 * depthMeter;
            ptrVert[off+1] = y0 * depthMeter;
            ptrVert[off+2] = depthMeter;
        }
    }

    return true;
}


Vec3f centroid(const cv::Mat &vertexMap)
{
    Vec3f centroid(0.0, 0.0, 0.0);

    size_t cnt = 0;
    for (int y = 0; y < vertexMap.rows; ++y)
    {
        for (int x = 0; x < vertexMap.cols; ++x)
        {
            cv::Vec3f pt = vertexMap.at<cv::Vec3f>(y, x);
            if (pt.val[2] > 0.0)
            {
                Vec3f pt3(pt.val[0], pt.val[1], pt.val[2]);
                centroid += pt3;
                ++cnt;
            }
        }
    }
    centroid /= float(cnt);

    return centroid;
}

// open3d::geometry::TriangleMesh CreateTriangleMesh(
//     const std::vector<Vec3f>& vertices, const std::vector<Vec3i>& faces) {
//     open3d::geometry::TriangleMesh mesh;
//     mesh.vertices_.resize(vertices.size());
//     mesh.triangles_.resize(faces.size());
//     for (size_t i = 0; i < vertices.size(); ++i) {
//         mesh.vertices_[i] = Eigen::Vector3d(vertices[i][0], vertices[i][1], vertices[i][2]);
//     }
//     for (size_t i = 0; i < faces.size(); ++i) {
//         mesh.triangles_[i] = Eigen::Vector3i(faces[i][0], faces[i][1], faces[i][2]);
//     }
//     mesh.ComputeVertexNormals();
//     return mesh;
// }

int main(int argc, char *argv[])
{
    // Default input sequence in folder
    // std::string dataFolder = std::string(STR(KILLINGFUSION_SOURCE_DIR)) + "/data/";
    std::string dataFolder = std::string(STR(KILLINGFUSION_SOURCE_DIR)) + "/data/Snoopy";

    // // Parse command line parameters
    // const char *params = 
    //     "{i|input| |input rgb-d sequence}"
    //     "{f|end|10000|last frame to process (0=all)}"
    //     "{n|iterations|100|max number of GD iterations}"
    //     "{a|alpha|0.1|Gradient Descent step size}"
    //     "{b|begin|0|First frame id to begin with}"
    //     "{d|debug|false|Debug mode}"
    //     "{wk|wk|0.5|Killing term weight}"
    //     "{ws|ws|0.1|Level set weight}"
    //     "{g|gamma|0.1|Killing property weight}";
    // cv::CommandLineParser cmd(argc, argv, params);

    // std::cout << "what are arguments" << std::endl;

    // cmd.printMessage();

    // int bValue = cmd.get<int>("b");
    // std::cout << "Value of parameter -b: " << bValue << std::endl;

    // // Input sequence
    // // Download from http://campar.in.tum.de/personal/slavcheva/deformable-dataset/index.html
    // std::string inputSequence = cmd.get<std::string>("input");

    // std::cout << "survived" << std::endl;

    // std::cout << inputSequence << std::endl;

    // if (inputSequence.empty())
    // {
    //     inputSequence = dataFolder;
    // }

    std::string inputSequence = dataFolder;
    std::cout << "input sequence: " << inputSequence << std::endl;

    // last frame Id
    // size_t lastFrameId = (size_t)cmd.get<int>("end");
    size_t lastFrameId = 22;
    std::cout << "Last Frame of sequence: " << lastFrameId << std::endl;

    // Max number of GD iterations
    // size_t iterations = (size_t)cmd.get<int>("iterations");
    size_t iterations = 1000;
    std::cout << "iterations: " << iterations << std::endl;

    // GD step size
    // float alpha = (float)cmd.get<float>("alpha");
    float alpha = 0.1;
    std::cout << "Gradient Descent step: " << alpha << std::endl;

    // First frame Id
    // size_t firstFrameId = (size_t)cmd.get<int>("begin");
    size_t firstFrameId = 2;
    std::cout << "First frame of sequence: " << firstFrameId << std::endl;

    // Debug mode
    // bool debugMode = (bool)cmd.get<bool>("debug");
    bool debugMode = false;
    std::cout << "Debug mode: " << debugMode << std::endl;

    // Killing term weight
    // float wk = (float)cmd.get<float>("wk");
    float wk = 0.1;
    std::cout << "w_k: " << wk << std::endl;

    // Level Set term weight
    // float ws = (float)cmd.get<float>("ws");
    float ws = 0.05;
    std::cout << "w_s: " << ws << std::endl;

    // Killing property term weight
    // float gamma = (float)cmd.get<float>("gamma");
    float gamma = 0.1;
    std::cout << "gamma: " << gamma << std::endl;

    //Trunction distance for the TSDF, which is also the scale for the tsdf gradients
    const float tsdfTruncationDistance = 0.05f;

    // Initialize cuda context
    cudaDeviceSynchronize(); CUDA_CHECK;

    // Load camera intrinsics
    Eigen::Matrix3f K;
    if (!loadIntrinsics(inputSequence + "/intrinsics_kinect1.txt", K))
    {
        std::cerr << "No intrinsics file found!" << std::endl;
        return 1;
    }
    std::cout << "K: " << std::endl << K << std::endl;

    // Create tsdf volume
    size_t gridW = 80, gridH = 80, gridD = 80;
    float voxelSize = 0.006;        // Voxel size in m
    Vec3i volDim(gridW, gridH, gridD);
    Vec3f volSize(gridW*voxelSize, gridH*voxelSize, gridD*voxelSize);
    TSDFVolume* tsdfGlobal = new TSDFVolume(volDim, volSize, K, tsdfTruncationDistance, 0);
    TSDFVolume* tsdfLive;
    // Initialize the deformation to zero initially
    float* deformationU = (float*)calloc(gridW*gridH*gridD, sizeof(float));
	float* deformationV = (float*)calloc(gridW*gridH*gridD, sizeof(float));
	float* deformationW = (float*)calloc(gridW*gridH*gridD, sizeof(float));

    for (size_t i = 0; i < gridW*gridH*gridD; i++)
    {
        deformationU[i] = 0.0f;
        deformationV[i] = 0.0f;
        deformationW[i] = 0.0f;
    }

    Optimizer* optimizer;

    // Create windows
    cv::namedWindow("color");
    cv::namedWindow("depth");
    cv::namedWindow("mask");

    // Process frames
    Mat4f poseVolume = Mat4f::Identity();
    cv::Mat color, depth, mask;


    // Create an Open3D visualizer window
    open3d::visualization::Visualizer visualizer;
    
    // visualizer.GetRenderOption().mesh_show_back_face_ = true;
    // auto visualizer = std::make_shared<open3d::visualization::Visualizer>();

    visualizer.CreateVisualizerWindow("Open3D Visualizer");
    visualizer.GetRenderOption().ToggleMeshShowBackFace();
    // visualizer->CreateVisualizerWindow("Open3D Visualizer");
    

    std::vector<Vec3f> current_vertices;
    std::vector<Vec3i> current_faces;

    // Add the mesh to the visualizer
    // open3d::geometry::TriangleMesh mesh ;
    // open3d::geometry::PointCloud mesh ;
    // auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();

    // visualizer->AddGeometry(std::make_shared<open3d::geometry::TriangleMesh>(mesh));
    // visualizer->AddGeometry(std::make_shared<open3d::geometry::TriangleMesh>(mesh));
    // visualizer->AddGeometry(mesh);
    // visualizer.AddGeometry(mesh);
    
    // visualizer->Run();

    MarchingCubes mc(volDim, volSize);
    float* tsdfGlobalAccumulated = (float*)calloc(gridW*gridH*gridD, sizeof(float));
    float* tsdfGlobalWeightsAccumulated = (float*)calloc(gridW*gridH*gridD, sizeof(float));

    for (size_t i = firstFrameId; i <= lastFrameId; ++i)
    {
        tsdfLive = new TSDFVolume(volDim, volSize, K, tsdfTruncationDistance, i);
        std::cout << "Working on frame: " << i;

        // Load input frame
        if (!loadFrame(inputSequence, i, color, depth, mask))
        {
            std::cerr << " ->Frame " << i << " could not be loaded!" << std::endl;
            break;
        }

        // Filter depth values outside of mask
        filterDepth(mask, depth);

        // Show input images
        cv::imshow("color", color);
        cv::imshow("depth", depth);
        cv::imshow("mask", mask);
        if (debugMode)
        {
            cv::waitKey(30);
        }
        

        // Get initial volume pose from centroid of first depth map
        if (i == firstFrameId)
        {
            // Initial pose for volume by computing centroid of first depth/vertex map
            cv::Mat vertMap;
            depthToVertexMap(K, depth, vertMap);
            Vec3f transCentroid = centroid(vertMap);
            poseVolume.topRightCorner<3,1>() = transCentroid;
            std::cout << std::endl << "pose centroid" << std::endl << poseVolume << std::endl;
			tsdfGlobal->integrate(poseVolume, color, depth);
            optimizer = new Optimizer(tsdfGlobal, deformationU, deformationV, deformationW, alpha, wk, ws, gamma, iterations, voxelSize, tsdfTruncationDistance, debugMode, gridW, gridH, gridD);

			
    		mc.computeIsoSurface(tsdfGlobal->ptrTsdf(), tsdfGlobal->ptrTsdfWeights(), tsdfGlobal->ptrColorR(), tsdfGlobal->ptrColorG(), tsdfGlobal->ptrColorB());

            current_vertices = mc.getVertices();
            current_faces = mc.getFaces();

            mesh->vertices_.resize(current_vertices.size());
            mesh->triangles_.resize(current_faces.size());

            for (size_t i = 0; i < current_vertices.size(); ++i) {
                mesh->vertices_[i] = Eigen::Vector3d(current_vertices[i][0], current_vertices[i][1], current_vertices[i][2]);
            }
            for (size_t i = 0; i < current_faces.size(); ++i) {
                mesh->triangles_[i] = Eigen::Vector3i(current_faces[i][0], current_faces[i][1], current_faces[i][2]);
            }
            mesh->ComputeVertexNormals();

            
            visualizer.AddGeometry(mesh);
            // visualizer.Run();
            
            // visualizer.Run();

			const std::string meshFilename = "./bin/result/mesh_canonical.ply";
			if (!mc.savePly(meshFilename))
			{
				std::cerr << "Could not save mesh!" << std::endl;
			}

        }
		else
		{
			// Integrate frame into tsdf volume
        	tsdfLive->integrate(poseVolume, color, depth);
            // Perform optimization

			optimizer->optimize(tsdfLive);

    		// mc.computeIsoSurface(tsdfLive->ptrTsdf(), tsdfLive->ptrTsdfWeights(), tsdfLive->ptrColorR(), tsdfLive->ptrColorG(), tsdfLive->ptrColorB());
            // mc.computeIsoSurface(tsdfGlobal->ptrTsdf(), tsdfGlobal->ptrTsdfWeights(), tsdfGlobal->ptrColorR(), tsdfGlobal->ptrColorG(), tsdfGlobal->ptrColorB());

            
            optimizer->getTSDFGlobalPtr(tsdfGlobalAccumulated);
            optimizer->getTSDFGlobalWeightsPtr(tsdfGlobalWeightsAccumulated);
            mc.computeIsoSurface(tsdfGlobalAccumulated, tsdfGlobalWeightsAccumulated, tsdfGlobal->ptrColorR(), tsdfGlobal->ptrColorG(), tsdfGlobal->ptrColorB());
			
            current_vertices = mc.getVertices();
            current_faces = mc.getFaces();

            // Print shape of vertices vector
            // Print shape of vertices vector
            // std::cout << "Shape of vertices: " << current_vertices.size() << " x " << (current_vertices.empty() ? 0 : current_vertices[0].size()) << std::endl;

            // // Print shape of faces vector
            // std::cout << "Shape of faces: " << current_faces.size() << " x " << (current_faces.empty() ? 0 : current_faces[0].size()) << std::endl;


            // mesh->points_.resize(current_vertices.size());
            //    for (size_t i = 0; i < current_vertices.size(); ++i) {
            //     mesh->points_[i] = Eigen::Vector3d(current_vertices[i][0], current_vertices[i][1], current_vertices[i][2]);
            // }

            // std::cout << mesh->HasPoints() << std::endl;
            // std::cout << "Minimum bound: " << mesh->GetMinBound().transpose() << std::endl;
            // std::cout << "Maximum bound: " << mesh->GetMaxBound().transpose() << std::endl;

            mesh->vertices_.resize(current_vertices.size());
            mesh->triangles_.resize(current_faces.size());

            for (size_t i = 0; i < current_vertices.size(); ++i) {
                mesh->vertices_[i] = Eigen::Vector3d(current_vertices[i][0], current_vertices[i][1], current_vertices[i][2]);
            }
            for (size_t i = 0; i < current_faces.size(); ++i) {
                mesh->triangles_[i] = Eigen::Vector3i(current_faces[i][0], current_faces[i][1], current_faces[i][2]);
            }
            mesh->ComputeVertexNormals();
            


            // mesh->CreateTriangleMesh(current_vertices, current_faces);

            // size_t num_vertices = mesh.vertices_.size();

            // std::cout << num_vertices << std::endl;

            // visualizer.UpdateGeometry(mesh);

            // visualizer.UpdateGeometry(std::make_shared<open3d::geometry::TriangleMesh>(mesh));
            // visualizer->Geometry();
            // visualizer->UpdateGeometry(mesh);
            // visualizer.Geometry();
            visualizer.UpdateGeometry();
            

            // Render the scene
            // visualizer->UpdateRender();
            visualizer.PollEvents();
            visualizer.UpdateRender();
            visualizer.ResetViewPoint();
            // visualizer->PollEvents();
            
            // std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Optional sleep to avoid high CPU usage
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            // cv::waitKey(30);

		}
        delete tsdfLive;
    }

    open3d::visualization::DrawGeometries({mesh}, "Point Cloud Example", 1920, 1080);
    

    optimizer->printTimes();
    // Extract mesh using marching cubes
    std::cout << "Extracting mesh..." << std::endl;
	// float* tsdfGlobalAccumulated = (float*)calloc(gridW*gridH*gridD, sizeof(float));
	// float* tsdfGlobalWeightsAccumulated = (float*)calloc(gridW*gridH*gridD, sizeof(float));
	optimizer->getTSDFGlobalPtr(tsdfGlobalAccumulated);
	optimizer->getTSDFGlobalWeightsPtr(tsdfGlobalWeightsAccumulated);
	MarchingCubes mcAcc(volDim, volSize);
	mcAcc.computeIsoSurface(tsdfGlobalAccumulated, tsdfGlobalWeightsAccumulated, tsdfGlobal->ptrColorR(), tsdfGlobal->ptrColorG(), tsdfGlobal->ptrColorB());
    // Save mesh
    std::cout << "Saving mesh..." << std::endl;
	const std::string meshAccFilename = "./bin/result/mesh_acc.ply";
	if (!mcAcc.savePly(meshAccFilename))
    {
        std::cerr << "Could not save accumulated mesh!" << std::endl;
    }

    // Clean up
    delete tsdfGlobal;
    if(optimizer) delete optimizer;

    cv::destroyAllWindows();

    return 0;
}