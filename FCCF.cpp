#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <chrono>
#include <algorithm>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/octree/octree.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transformation_from_correspondences.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>
#include <ceres/problem.h>
// #include <teaser/ply_io.h>
// #include <teaser/registration.h>
// #include <teaser/matcher.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>

typedef struct voxelnode
{
	float centry_x;
	float centry_y;
	float centry_z;
	float normal_x;
	float normal_y;
	float normal_z;
	int voxel_point_size;
	bool is_allocate;
	pcl::FPFHSignature33 fpfh_value;
	pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud_ptr;
}voxelnode;

typedef struct facenode
{
	float average_centry_x;
	float average_centry_y;
	float average_centry_z;
	float average_normal_x;
	float average_normal_y;
	float average_normal_z;
	float face_point_size;
	bool is_allocate;
	std::vector<voxelnode> voxelgrothnode;
}facenode;

typedef struct face_base
{
	int index1;
	int index2;
	float angel;
}face_base;

typedef struct face_three
{
	int index1;
	int index2;
	int index3;
}face_three;

typedef struct transform_q_t
{
	float qw;
	float qx;
	float qy;
	float qz;
	float tx;
	float ty;
	float tz;
	bool is_allocate;		
}transform_q_t;

typedef struct transform_score
{
	Eigen::Matrix4f transformation_matrix;
	float score;	
	float score2;
}transform_score;

typedef struct pair_face
{
	int index1;
	int index2;
	float important;
	Eigen::Vector3f point1;
	Eigen::Vector3f normal1;
	Eigen::Vector3f point2;
	Eigen::Vector3f normal2;
}pair_face;

typedef struct pair_point
{
	Eigen::Vector3f point1;
	Eigen::Vector3f point2;
}pair_point;

typedef struct high_score
{
	transform_q_t qt;
	float score;
}high_score;

typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandlerT;
typedef pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> RandomRGB;


//parameter for plane feature extraction and fusion
//l and k
// float parameter_l1=0.5; // 0.5 (ori)
// float parameter_l2=1.0; // 1.0 (ori)
// float parameter_k1=5.0; // 5.0 (ori)
// float parameter_k2=2.0; // 2.0 (ori)
float parameter_l1=0.5; // 0.5
float parameter_l2=1.0; // 1.0
float parameter_k1=5.0; // 5.0
float parameter_k2=2.0; // 2.0 
//threshold of normal vector
float normal_vector_threshold1=5.0; // 5.0
float normal_vector_threshold2=8.0; // 8.0
//voxel size
float face_voxel_size=1.0; // 1.0
//if the point in voxel is less than this value, it will not be processed
float voxel_point_threshold=5; // 5
//threshold of plane curvature
float curvature_threshold=0.05;
//select the number or size of planes
//float select_plane_size=0;
float select_plane_number=15; // qs (ori)
// float select_plane_number=20; // qs

//parameter for quick verify
float quick_verify_angel_threshold=10.0;
float quick_verify_distance_threshold=2.0;
float required_optimize_plane=4.0;

//parameter for fine verify
float fine_verify_voxel_size=0.5;
float fine_verify_number=4; // qv 10 (ori)
// float fine_verify_number=50; // qv 10

//parameter for calculate transform matrix
//Included angle of plane threshold
float included_angle_same_threshold=5.0;
float included_angle_min_threshold=30.0;
float included_angle_max_threshold=150.0;
//threshold for select the third plane
float third_plane_threshold=0.5;
//third plane normal vector angle difference threshold
float third_plane_normal_threshold=5.0;

//parameter for transform matrix cluster
//ransform matrix below the threshold are not clustered
float cluster_number_threshold=10;
float cluster_angel_threshold=2.0;
float cluster_distance_threshold=0.8;
//select the size or number of clusters
//float seclct_cluster_size=0;
float seclct_cluster_number=200; // qc 200 (ori)
// float seclct_cluster_number=500; // qc 200

// roughness
float rough_threshold_gl = 2; // in select_base()
float LeafSize;

namespace ceres {
	struct LidarPlaneFactor
	{
		LidarPlaneFactor(Eigen::Vector3f normal1_,Eigen::Vector3f point1_,Eigen::Vector3f normal2_,Eigen::Vector3f point2_,float important_):normal1(normal1_),point1(point1_),normal2(normal2_),point2(point2_),important(important_){}
		template <typename T>
		bool operator()(const T *q, const T *t, T *residual) const
		{
			Eigen::Matrix<T, 3, 1> n1{T(normal1[0]), T(normal1[1]), T(normal1[2])};
			Eigen::Matrix<T, 3, 1> n2{T(normal2[0]), T(normal2[1]), T(normal2[2])};
			Eigen::Matrix<T, 3, 1> p1{T(point1[0]), T(point1[1]), T(point1[2])};
			Eigen::Matrix<T, 3, 1> p2{T(point2[0]), T(point2[1]), T(point2[2])};			
			Eigen::Quaternion<T> Q{q[3], q[0], q[1], q[2]};
			Eigen::Matrix<T, 3, 1> tf{t[0],t[1],t[2]};
			n2=Q*n2;
			p2=Q*p2+tf;
			residual[0] = T(important)*(n1.cross(n2)).norm();
			residual[1] = T(important)*sqrt(((n1.dot(p1))-(n2.dot(p2)))*((n1.dot(p1))-(n2.dot(p2))));
			return true;
		}

		static ceres::CostFunction *Create(const Eigen::Vector3f normal1_,const Eigen::Vector3f point1_,const Eigen::Vector3f normal2_,const Eigen::Vector3f point2_,const float important_)
		{
			return (new ceres::AutoDiffCostFunction<
					LidarPlaneFactor, 2, 4, 3>(
				new LidarPlaneFactor(normal1_,point1_,normal2_,point2_,important_)));
		}

		Eigen::Vector3f normal1,point1,normal2,point2;
		float important;
	};
}

void ceres_refine(Eigen::Matrix4f &new_transformation_matrix,std::vector<pair_face> &pair_face_vecter)
{
	double para_q[4] = {0, 0, 0, 1};
	double para_t[3] = {0, 0, 0};
	ceres::LocalParameterization *q_parameterization =
		new ceres::EigenQuaternionParameterization();
	ceres::Problem::Options problem_options;
	ceres::Problem problem(problem_options);
	problem.AddParameterBlock(para_q, 4, q_parameterization);
	problem.AddParameterBlock(para_t, 3);
	for(auto it1=pair_face_vecter.begin();it1!=pair_face_vecter.end();it1++)
	{
		ceres::CostFunction *cost_function = ceres::LidarPlaneFactor::Create((*it1).normal1,(*it1).point1,(*it1).normal2,(*it1).point2,(*it1).important);
		problem.AddResidualBlock(cost_function, nullptr, para_q, para_t);
	}		
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.max_num_iterations = 50;
	options.minimizer_progress_to_stdout = false;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	Eigen::Quaternionf	quaterniond_temp;
	quaterniond_temp.w()=para_q[3];
	quaterniond_temp.x()=para_q[0];
	quaterniond_temp.y()=para_q[1];
	quaterniond_temp.z()=para_q[2];
	Eigen::Matrix3f R = quaterniond_temp.toRotationMatrix();
	new_transformation_matrix(0,0)=R(0,0);
	new_transformation_matrix(0,1)=R(0,1);
	new_transformation_matrix(0,2)=R(0,2);
	new_transformation_matrix(1,0)=R(1,0);
	new_transformation_matrix(1,1)=R(1,1);
	new_transformation_matrix(1,2)=R(1,2);
	new_transformation_matrix(2,0)=R(2,0);
	new_transformation_matrix(2,1)=R(2,1);
	new_transformation_matrix(2,2)=R(2,2);
	new_transformation_matrix(0,3)=para_t[0];
	new_transformation_matrix(1,3)=para_t[1];
	new_transformation_matrix(2,3)=para_t[2];
}

void showDVector(std::vector<double> &vec){
	for(auto it = vec.begin(); it != vec.end(); it++)
		std::cout << (*it) << " ";
	std::cout << std::endl;
}

void showPic(std::vector<facenode> &source){
	pcl::visualization::PCLVisualizer viewer("Picture");
	viewer.setBackgroundColor(255, 255, 255);
	int index  = 0;
	pcl::PointCloud<pcl::Normal>::Ptr face_normal(new pcl::PointCloud<pcl::Normal>); // Planes' normal vectors
	pcl::PointCloud<pcl::PointXYZ>::Ptr face_centroid(new pcl::PointCloud<pcl::PointXYZ>); // Planes' centroids
	for(auto it1 = source.begin(); it1 != source.end() ; it1++){
		pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud_ptr_temp(new pcl::PointCloud<pcl::PointXYZ>);

		(*face_normal).push_back(pcl::Normal((*it1).average_normal_x, (*it1).average_normal_y, (*it1).average_normal_z));
		(*face_centroid).push_back(pcl::PointXYZ((*it1).average_centry_x, (*it1).average_centry_y, (*it1).average_centry_z));

		std::string name = std::to_string(index);
		name = "pt" + name;
		for(auto it2 = (*it1).voxelgrothnode.begin(); it2 != (*it1).voxelgrothnode.end(); it2++)
		{
			*voxel_cloud_ptr_temp += *((*it2).voxel_cloud_ptr);
		}
		viewer.addPointCloud (voxel_cloud_ptr_temp,RandomRGB(voxel_cloud_ptr_temp),name);
		// viewer.addPointCloud (voxel_cloud_ptr_temp, ColorHandlerT(voxel_cloud_ptr_temp, 0, 255, 0), name);
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, name);
	 	index++;
	}

	viewer.addPointCloud (face_centroid,ColorHandlerT(face_centroid, 255, 0, 0),"centroid");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "centroid");
	for(int i = 0 ; i < int((*face_centroid).size()) ; i++) // show normal vectors
		viewer.addArrow<pcl::PointXYZ> ( pcl::PointXYZ((*face_centroid)[i].x + (*face_normal)[i].normal_x, (*face_centroid)[i].y + (*face_normal)[i].normal_y, (*face_centroid)[i].z + (*face_normal)[i].normal_z), pcl::PointXYZ((*face_centroid)[i].x, (*face_centroid)[i].y, (*face_centroid)[i].z), 255, 0, 0, false, std::to_string(i));
	viewer.addCoordinateSystem (1.0);
	// viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(face_centroid,face_normal,1,0.5,"normals");
	viewer.spin();
}

void showPointPic(facenode source){
	pcl::visualization::PCLVisualizer viewer("Picture");
	viewer.setBackgroundColor(255, 255, 255);
	int index  = 0;
	pcl::PointCloud<pcl::Normal>::Ptr voxel_normal(new pcl::PointCloud<pcl::Normal>); // Voxels' normal vectors
	pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_centroid(new pcl::PointCloud<pcl::PointXYZ>); // Voxels' centroids
	int aindex = 0;
	pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud_ptr_temp(new pcl::PointCloud<pcl::PointXYZ>);

	// (*voxel_centroid).push_back(pcl::PointXYZ((*it1).average_centry_x, (*it1).average_centry_y, (*it1).average_centry_z));
	RandomRGB rgb(voxel_cloud_ptr_temp);
	std::string name = std::to_string(index);
	name = "pt" + name;
	for(auto it2 = source.voxelgrothnode.begin(); it2 != source.voxelgrothnode.end(); it2++)
	{
		*voxel_cloud_ptr_temp += *((*it2).voxel_cloud_ptr);
		viewer.addArrow<pcl::PointXYZ> ( pcl::PointXYZ((*it2).centry_x + (*it2).normal_x, (*it2).centry_y + (*it2).normal_y, (*it2).centry_z + (*it2).normal_z), pcl::PointXYZ((*it2).centry_x, (*it2).centry_y, (*it2).centry_z), 255, 0, 0, false, std::to_string(aindex));
		// (*voxel_normal).push_back(pcl::Normal((*it2).normal_x, (*it2).normal_y, (*it2).normal_z));
		aindex++;
	}
	viewer.addPointCloud (voxel_cloud_ptr_temp,rgb,name);
	// viewer.addPointCloud (voxel_cloud_ptr_temp, ColorHandlerT(voxel_cloud_ptr_temp, 0, 255, 0), name);
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, name);
	index++;

	// Show arrows of normal vectors
	// viewer.addPointCloud (voxel_centroid,ColorHandlerT(voxel_centroid, 255, 0, 0),"centroid");
	// viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "centroid");
	// for(int i = 0 ; i < (*voxel_centroid).size() ; i++) // show normal vectors
	// 	viewer.addArrow<pcl::PointXYZ> ( pcl::PointXYZ((*voxel_centroid)[i].x + (*voxel_normal)[i].normal_x, (*voxel_centroid)[i].y + (*voxel_normal)[i].normal_y, (*voxel_normal)[i].z + (*voxel_normal)[i].normal_z), pcl::PointXYZ((*voxel_centroid)[i].x, (*voxel_centroid)[i].y, (*voxel_centroid)[i].z), 255, 0, 0, false, std::to_string(i));
	// viewer.addCoordinateSystem (1.0);
	// viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(face_centroid,face_normal,1,0.5,"normals");
	viewer.spin();
}

void average_normal(Eigen::Vector3f &rotaionvector1,Eigen::Vector3f &rotaionvector2,std::vector<transform_q_t> &matrixvector)
{
	float sumnx1=0;
	float sumny1=0;
	float sumnz1=0;
	float sumnx2=0;
	float sumny2=0;
	float sumnz2=0;	
	for(auto it1=matrixvector.begin();it1!=matrixvector.end();it1++)
	{
		Eigen::Vector3f n1=Eigen::Vector3f(1,0,0);
		Eigen::Vector3f n2=Eigen::Vector3f(0,1,0);
		Eigen::Quaternionf Q;
		Q.w()=(*it1).qw;
		Q.x()=(*it1).qx;
		Q.y()=(*it1).qy;
		Q.z()=(*it1).qz;
		n1=Q*n1;
		n2=Q*n2;
		sumnx1=sumnx1+n1[0];
		sumny1=sumny1+n1[1];
		sumnz1=sumnz1+n1[2];
		sumnx2=sumnx2+n2[0];
		sumny2=sumny2+n2[1];
		sumnz2=sumnz2+n2[2];		
	}
	float averagenx1=sumnx1/matrixvector.size();
	float averageny1=sumny1/matrixvector.size();
	float averagenz1=sumnz1/matrixvector.size();
	float averagenx2=sumnx2/matrixvector.size();
	float averageny2=sumny2/matrixvector.size();
	float averagenz2=sumnz2/matrixvector.size();	
	Eigen::Vector3f averagen1=Eigen::Vector3f(averagenx1,averageny1,averagenz1);
	Eigen::Vector3f averagen2=Eigen::Vector3f(averagenx2,averageny2,averagenz2);
	averagen1.normalize();
	averagen2.normalize();
	rotaionvector1[0]=averagen1[0];
	rotaionvector1[1]=averagen1[1];
	rotaionvector1[2]=averagen1[2];
	rotaionvector2[0]=averagen2[0];
	rotaionvector2[1]=averagen2[1];
	rotaionvector2[2]=averagen2[2];	
}

float compute_normal_angel(float normal_x1,float normal_y1,float normal_z1,float normal_x2,float normal_y2,float normal_z2)
{
	Eigen::Vector3d n1=Eigen::Vector3d(normal_x1,normal_y1,normal_z1);
	Eigen::Vector3d n2=Eigen::Vector3d(normal_x2,normal_y2,normal_z2);
    float n1n3=n1.transpose()*n2;
    float cos_theta=n1n3/((n1.norm())*(n2.norm()));
    float theta=acos(cos_theta)*180/M_PI;
	return theta;
}

bool compare_normal(float normal_x1,float normal_y1,float normal_z1,float normal_x2,float normal_y2,float normal_z2,float normal_vector_threshold)
{
	float theta=compute_normal_angel(normal_x1,normal_y1,normal_z1,normal_x2,normal_y2,normal_z2);
	if(theta>normal_vector_threshold)
	{
		return false;
	}else
	{
		return true;
	}
}

bool compare_plane(float normal_x1,float normal_y1,float normal_z1,float centry_x1,float centry_y1,float centry_z1,float normal_x2,float normal_y2,float normal_z2,float centry_x2,float centry_y2,float centry_z2,float parameter_l,float parameter_k)
{
	Eigen::Vector3d n1=Eigen::Vector3d(normal_x1,normal_y1,normal_z1);
	Eigen::Vector3d n2=Eigen::Vector3d(normal_x2,normal_y2,normal_z2);
	float vectorlength= sqrt((centry_x1-centry_x2)*(centry_x1-centry_x2)+(centry_y1-centry_y2)*(centry_y1-centry_y2)+(centry_z1-centry_z2)*(centry_z1-centry_z2));
	Eigen::Vector3d n3=Eigen::Vector3d((centry_x1-centry_x2)/vectorlength,(centry_y1-centry_y2)/vectorlength,(centry_z1-centry_z2)/vectorlength);
	float n1n3=fabs(n1.transpose()*n3);
	float n2n3=fabs(n2.transpose()*n3);
	float co_plane_threash=parameter_l/(parameter_k*vectorlength+1);
	if(n1n3<co_plane_threash && n2n3<co_plane_threash)
	{
		return true;
	}else
	{
		return false;
	}
}

void range_face(std::vector<facenode> &face_vecter)
{
	for(auto it1=face_vecter.begin();it1!=face_vecter.end();it1++)
	{
		if(it1!=(face_vecter.end()-1))
		{
			for(auto it2=it1+1;it2!=face_vecter.end();it2++)
			{
				if((*it1).voxelgrothnode.size()<(*it2).voxelgrothnode.size())
				{
					facenode facenodetemp;
					facenodetemp=(*it1);
					(*it1)=(*it2);
					(*it2)=facenodetemp;
				}
			}
		}
	}
}

void select_base(std::vector<face_base> &base_vecter, std::vector<facenode> &face_vecter, std::vector<int> &type_index, std::vector<double> &new_theta_vector)
{
	float threshold_min = included_angle_min_threshold;
	float threshold_max = included_angle_max_threshold;
	int index1 = 0;
	double theta_threshold1 =  rough_threshold_gl;

	for (auto it1 = face_vecter.begin(); it1 != face_vecter.end(); it1++)
	{
		int index2 = 0;
		for (auto it2 = face_vecter.begin(); it2 != face_vecter.end(); it2++)
		{
			if (index1 < index2)
			{
				float angel = compute_normal_angel((*it1).average_normal_x, (*it1).average_normal_y, (*it1).average_normal_z, (*it2).average_normal_x, (*it2).average_normal_y, (*it2).average_normal_z);
				if (threshold_min < angel && angel < threshold_max)
				{
					face_base face_base_temp;
					face_base_temp.index1 = index1;
					face_base_temp.index2 = index2;
					face_base_temp.angel = angel;
					base_vecter.push_back(face_base_temp);
					// ---- Record theta -----
					// both smooth -> 0 , both rough -> 1 , one smooth + one rough -> 2
					////////////----------------////////////////////
					if(new_theta_vector[index1] <= theta_threshold1 && new_theta_vector[index2] <= theta_threshold1)
						type_index.push_back(0);
					else if(new_theta_vector[index1] > theta_threshold1 && new_theta_vector[index2] > theta_threshold1)
						type_index.push_back(1);
					else if(new_theta_vector[index1] <= theta_threshold1 && new_theta_vector[index2] > theta_threshold1)
						type_index.push_back(2);
					else if(new_theta_vector[index1] > theta_threshold1 && new_theta_vector[index2] <= theta_threshold1)
						type_index.push_back(2);
				}
			}
			index2++;
		}
		index1++;
	}
}

void face_extrate(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src,std::vector<facenode> &face_vecter,pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sub, std::vector<double> &theta_vector)
{	
	Eigen::Matrix<float, 4, 1> cloud_src_centroid; 
	pcl::compute3DCentroid(*cloud_src, cloud_src_centroid);
	float resolution = face_voxel_size;
	pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
	octree.setInputCloud(cloud_src);
	octree.addPointsFromInputCloud();
	pcl::octree::OctreePointCloud<pcl::PointXYZ>::AlignedPointTVector vec;
	octree.getOccupiedVoxelCenters(vec);
	std::vector<voxelnode> voxel_vector;
	for(int i=0;i<int(vec.size());i++)
	{
		std::vector<int> pointIdxVec;
		if (octree.voxelSearch ((*(vec.begin()+i)), pointIdxVec))
		{
			if(pointIdxVec.size() > voxel_point_threshold)
			{
				pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne_src;
				Eigen::Matrix<float, 4, 1> centroid;
				pcl::compute3DCentroid((*cloud_src),pointIdxVec,centroid);
				float nx;
				float ny;
				float nz;
				float curvature;
                ne_src.computePointNormal((*cloud_src),pointIdxVec,nx,ny,nz,curvature);

				if(curvature<curvature_threshold)
				{
					voxelnode voxelnode_temp;
					voxelnode_temp.centry_x=centroid[0];
					voxelnode_temp.centry_y=centroid[1];
					voxelnode_temp.centry_z=centroid[2];
					voxelnode_temp.voxel_point_size=pointIdxVec.size();
					Eigen::Vector3f to_centry=Eigen::Vector3f((centroid[0]-cloud_src_centroid[0]),(centroid[1]-cloud_src_centroid[1]),(centroid[2]-cloud_src_centroid[2]));
					Eigen::Vector3f normal_vector=Eigen::Vector3f(nx,ny,nz);
					if((to_centry.dot(normal_vector))<0)
					{
						voxelnode_temp.normal_x=nx;
						voxelnode_temp.normal_y=ny;
						voxelnode_temp.normal_z=nz;
					}else
					{
						voxelnode_temp.normal_x=-nx;
						voxelnode_temp.normal_y=-ny;
						voxelnode_temp.normal_z=-nz;						
					}
					voxelnode_temp.is_allocate=false;
					pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud_ptr_temp(new pcl::PointCloud<pcl::PointXYZ>);
					for(auto it=pointIdxVec.begin();it!=pointIdxVec.end();it++)
					{
						(*voxel_cloud_ptr_temp).push_back((*cloud_src)[*it]);
					}
					voxelnode_temp.voxel_cloud_ptr=voxel_cloud_ptr_temp;
					voxel_vector.push_back(voxelnode_temp);
				}else
				{
					for (auto it = pointIdxVec.begin(); it != pointIdxVec.end(); it++)
					{
						(*cloud_sub).push_back((*cloud_src)[*it]);
					}
				}
			}			
		}
	}

    std::vector<facenode> voxel_vector_groth;
	for(auto it1=voxel_vector.begin();it1!=voxel_vector.end();it1++)
	{
		if((*it1).is_allocate==false)
		{
			facenode facenode_temp;
			(*it1).is_allocate=true;
			facenode_temp.voxelgrothnode.push_back(*it1);
			facenode_temp.face_point_size=(*it1).voxel_point_size;
			facenode_temp.average_normal_x=(*it1).normal_x;
			facenode_temp.average_normal_y=(*it1).normal_y;
			facenode_temp.average_normal_z=(*it1).normal_z;	
			facenode_temp.average_centry_x=(*it1).centry_x;
			facenode_temp.average_centry_y=(*it1).centry_y;
			facenode_temp.average_centry_z=(*it1).centry_z;
			for(auto it2=voxel_vector.begin();it2!=voxel_vector.end();it2++)
			{
				if((*it2).is_allocate==false)
				{
					bool is_same=true;
					bool is_coplane=true;
					is_same=compare_normal(facenode_temp.average_normal_x,facenode_temp.average_normal_y,facenode_temp.average_normal_z,(*it2).normal_x,(*it2).normal_y,(*it2).normal_z,normal_vector_threshold1);
					is_coplane=compare_plane(facenode_temp.average_normal_x,facenode_temp.average_normal_y,facenode_temp.average_normal_z,facenode_temp.average_centry_x,facenode_temp.average_centry_y,facenode_temp.average_centry_z,(*it2).normal_x,(*it2).normal_y,(*it2).normal_z,(*it2).centry_x,(*it2).centry_y,(*it2).centry_z,parameter_l1,parameter_k1);
					if(is_same==true && is_coplane==true)
					{
						facenode_temp.voxelgrothnode.push_back(*it2);
						(*it2).is_allocate=true;
						float sum_voxel_point_size=0;
						float average_centry_x=0;
						float average_centry_y=0;
						float average_centry_z=0;
						float average_normal_x=0;
						float average_normal_y=0;
						float average_normal_z=0;			
						for(auto it3=(facenode_temp.voxelgrothnode).begin();it3!=(facenode_temp.voxelgrothnode).end();it3++)
						{
							sum_voxel_point_size=sum_voxel_point_size+((*it3).voxel_point_size);
							average_centry_x=average_centry_x+((*it3).centry_x)*((*it3).voxel_point_size);
							average_centry_y=average_centry_y+((*it3).centry_y)*((*it3).voxel_point_size);
							average_centry_z=average_centry_z+((*it3).centry_z)*((*it3).voxel_point_size);
							average_normal_x=average_normal_x+((*it3).normal_x)*((*it3).voxel_point_size);
							average_normal_y=average_normal_y+((*it3).normal_y)*((*it3).voxel_point_size);
							average_normal_z=average_normal_z+((*it3).normal_z)*((*it3).voxel_point_size);	
						}
						facenode_temp.face_point_size=sum_voxel_point_size;
						facenode_temp.average_centry_x=average_centry_x/sum_voxel_point_size;
						facenode_temp.average_centry_y=average_centry_y/sum_voxel_point_size;
						facenode_temp.average_centry_z=average_centry_z/sum_voxel_point_size;
						facenode_temp.average_normal_x=average_normal_x/sum_voxel_point_size;
						facenode_temp.average_normal_y=average_normal_y/sum_voxel_point_size;
						facenode_temp.average_normal_z=average_normal_z/sum_voxel_point_size;
					}					
				}
			}	
			facenode_temp.is_allocate=false;
			voxel_vector_groth.push_back(facenode_temp);
		}  
	}

    for(auto it1=voxel_vector_groth.begin();it1!=voxel_vector_groth.end();it1++)
	{	
		if((*it1).is_allocate==false)
		{
			bool newadd=true;
			while(newadd==true)
			{
				newadd=false;
				for(auto it2=voxel_vector_groth.begin();it2!=voxel_vector_groth.end();it2++)
				{
					if((it2)!=(it1) && (*it2).is_allocate==false)
					{
						bool is_same=true;
						bool is_coplane=true;
						is_same=compare_normal((*it1).average_normal_x,(*it1).average_normal_y,(*it1).average_normal_z,(*it2).average_normal_x,(*it2).average_normal_y,(*it2).average_normal_z,normal_vector_threshold2);
						is_coplane=compare_plane((*it1).average_normal_x,(*it1).average_normal_y,(*it1).average_normal_z,(*it1).average_centry_x,(*it1).average_centry_y,(*it1).average_centry_z,(*it2).average_normal_x,(*it2).average_normal_y,(*it2).average_normal_z,(*it2).average_centry_x,(*it2).average_centry_y,(*it2).average_centry_z,parameter_l2,parameter_k2);
						if(is_same==true && is_coplane==true)
						{
							newadd=true;
							(*it2).is_allocate=true;
							for(auto it4=((*it2).voxelgrothnode).begin();it4!=((*it2).voxelgrothnode).end();it4++)
							{
								((*it1).voxelgrothnode).push_back(*it4);
							}
							float sum_voxel_point_size=0;
							float average_centry_x=0;
							float average_centry_y=0;
							float average_centry_z=0;
							float average_normal_x=0;
							float average_normal_y=0;
							float average_normal_z=0;			
							for(auto it3=((*it1).voxelgrothnode).begin();it3!=((*it1).voxelgrothnode).end();it3++)
							{
								sum_voxel_point_size=sum_voxel_point_size+((*it3).voxel_point_size);
								average_centry_x=average_centry_x+((*it3).centry_x)*((*it3).voxel_point_size);
								average_centry_y=average_centry_y+((*it3).centry_y)*((*it3).voxel_point_size);
								average_centry_z=average_centry_z+((*it3).centry_z)*((*it3).voxel_point_size);
								average_normal_x=average_normal_x+((*it3).normal_x)*((*it3).voxel_point_size);
								average_normal_y=average_normal_y+((*it3).normal_y)*((*it3).voxel_point_size);
								average_normal_z=average_normal_z+((*it3).normal_z)*((*it3).voxel_point_size);	
							}
							(*it1).face_point_size=sum_voxel_point_size;
							(*it1).average_centry_x=average_centry_x/sum_voxel_point_size;
							(*it1).average_centry_y=average_centry_y/sum_voxel_point_size;
							(*it1).average_centry_z=average_centry_z/sum_voxel_point_size;
							(*it1).average_normal_x=average_normal_x/sum_voxel_point_size;
							(*it1).average_normal_y=average_normal_y/sum_voxel_point_size;
							(*it1).average_normal_z=average_normal_z/sum_voxel_point_size;
						}					
					}
				}
			}
		}
	}

	range_face(voxel_vector_groth);
	
	std::vector<facenode> face_vecter_chose;
	int currentSelectNum = 0;
	int currentVectorIndex = 0;
	for(auto it1=voxel_vector_groth.begin();it1!=voxel_vector_groth.end();it1++)
	{
		if((*it1).is_allocate==false)
		{
			face_vecter_chose.push_back((*it1));
            double thetaSum = 0;
            // check if planar
            for(int i = 0 ; i < int(it1->voxelgrothnode.size()); i++){
                double theta=compute_normal_angel(it1->average_normal_x,it1->average_normal_y,it1->average_normal_z,it1->voxelgrothnode[i].normal_x,it1->voxelgrothnode[i].normal_y,it1->voxelgrothnode[i].normal_z);
                thetaSum += abs(theta);
            }
			thetaSum /= it1->voxelgrothnode.size();
			theta_vector.push_back(thetaSum);
			currentSelectNum++;
		}
		if(currentSelectNum > select_plane_number)
		{
			break;
		}
		currentVectorIndex++;		
	}
	
	face_vecter.swap(face_vecter_chose);
}

float quick_verify(Eigen::Matrix4f &transformation_matrix, std::vector<facenode> &face_vecter1, std::vector<facenode> &face_vecter2)
{
	int faces_size1 = 0;
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normal1(new pcl::PointCloud<pcl::PointNormal>);
	for (auto it1 = face_vecter1.begin(); it1 != face_vecter1.end(); it1++)
	{
		pcl::PointNormal normal_temp;
		normal_temp.x = (*it1).average_centry_x;
		normal_temp.y = (*it1).average_centry_y;
		normal_temp.z = (*it1).average_centry_z;
		normal_temp.normal_x = (*it1).average_normal_x;
		normal_temp.normal_y = (*it1).average_normal_y;
		normal_temp.normal_z = (*it1).average_normal_z;
		faces_size1 = faces_size1 + (*it1).face_point_size;
		(*cloud_normal1).push_back(normal_temp);
	}
	int faces_size2 = 0;
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normal2(new pcl::PointCloud<pcl::PointNormal>);
	for (auto it1 = face_vecter2.begin(); it1 != face_vecter2.end(); it1++)
	{
		pcl::PointNormal normal_temp;
		normal_temp.x = (*it1).average_centry_x;
		normal_temp.y = (*it1).average_centry_y;
		normal_temp.z = (*it1).average_centry_z;
		normal_temp.normal_x = (*it1).average_normal_x;
		normal_temp.normal_y = (*it1).average_normal_y;
		normal_temp.normal_z = (*it1).average_normal_z;
		faces_size2 = faces_size2 + (*it1).face_point_size;
		(*cloud_normal2).push_back(normal_temp);
	}

	std::vector<pair_face> pair_face_vecter;
	pcl::transformPointCloudWithNormals(*cloud_normal2, *cloud_normal2, transformation_matrix);

	int index_first = 0;
	for (const auto &point1 : *cloud_normal1)
	{
		std::vector<int> candidate_vecter;
		bool find_coplane = false;
		int index_second = 0;
		for (const auto &point2 : *cloud_normal2)
		{		
			float angel = compute_normal_angel(point1.normal_x, point1.normal_y, point1.normal_z, point2.normal_x, point2.normal_y, point2.normal_z);
			Eigen::Vector3d n1=Eigen::Vector3d(point1.normal_x,point1.normal_y,point1.normal_z);
			Eigen::Vector3d n2=Eigen::Vector3d(point2.normal_x,point2.normal_y,point2.normal_z);
			Eigen::Vector3d c1=Eigen::Vector3d(point1.x,point1.y,point1.z);
			Eigen::Vector3d c2=Eigen::Vector3d(point2.x,point2.y,point2.z);		
			float distance1=n1.dot(c1);
			float distance2=n2.dot(c2);
         	float distance=fabs(distance1-distance2);
			if (angel<quick_verify_angel_threshold && distance<quick_verify_distance_threshold)
			{
				find_coplane = true;
				candidate_vecter.push_back(index_second);
			}
			index_second++;
		}

		float size1 = face_vecter1[index_first].face_point_size;
		int best_candidate = 0;
		float best_candidate_important=0;
		float best_candidate_score = 0;
		for (auto it1 = candidate_vecter.begin(); it1 != candidate_vecter.end(); it1++)
		{
			float size2 = face_vecter2[(*it1)].face_point_size;
         	float min=size1<size2?size1:size2;
         	float max=size1>size2?size1:size2;
			float candidate_score = min/max;
         	float candidate_important=(2*min)/(faces_size1 + faces_size2);
			if (candidate_score > best_candidate_score)
			{
				best_candidate_important = candidate_important;
				best_candidate_score = candidate_score;
				best_candidate = (*it1);
			}
		}
		if (find_coplane == true)
		{
			pair_face pair_face_temp;
			pair_face_temp.index1 = index_first;
			pair_face_temp.index2 = best_candidate;
			pair_face_temp.important =best_candidate_important;
			pair_face_temp.point1 = Eigen::Vector3f((*cloud_normal1)[index_first].x, (*cloud_normal1)[index_first].y, (*cloud_normal1)[index_first].z);
			pair_face_temp.normal1 = Eigen::Vector3f((*cloud_normal1)[index_first].normal_x, (*cloud_normal1)[index_first].normal_y, (*cloud_normal1)[index_first].normal_z);
			pair_face_temp.point2 = Eigen::Vector3f((*cloud_normal2)[best_candidate].x, (*cloud_normal2)[best_candidate].y, (*cloud_normal2)[best_candidate].z);
			pair_face_temp.normal2 = Eigen::Vector3f((*cloud_normal2)[best_candidate].normal_x, (*cloud_normal2)[best_candidate].normal_y, (*cloud_normal2)[best_candidate].normal_z);
			pair_face_vecter.push_back(pair_face_temp);
		}
		index_first++;
	}

	Eigen::Matrix4f new_transformation_matrix = Eigen::Matrix4f::Identity();
	if (pair_face_vecter.size() >= required_optimize_plane)
	{
		ceres_refine(new_transformation_matrix, pair_face_vecter);
		transformation_matrix = new_transformation_matrix * transformation_matrix;
	}
	float score = 0;
	for (auto it1 = pair_face_vecter.begin(); it1 != pair_face_vecter.end(); it1++)
	{
		score = score + (*it1).important;
	}
	return score;
}

float fine_verify(Eigen::Matrix4f &transformation_matrix,pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_source,pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_target)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudecorlor(new pcl::PointCloud<pcl::PointXYZRGB>);		
	pcl::transformPointCloud (*cloud_target, *cloud_target, transformation_matrix);
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_fuse(new pcl::PointCloud<pcl::PointXYZI>);
	*cloud_fuse=*cloud_source+*cloud_target;		
	float resolution = fine_verify_voxel_size;
	pcl::octree::OctreePointCloudSearch<pcl::PointXYZI> octree(resolution);
	octree.setInputCloud(cloud_fuse);
	octree.addPointsFromInputCloud();
	pcl::octree::OctreePointCloud<pcl::PointXYZI>::AlignedPointTVector vec;
	octree.getOccupiedVoxelCenters(vec);
	float threash_num=1;
	float similar_num=0;
	float allinvec=0;
	for(int i=0;i<int(vec.size());i++)
	{
		std::vector<int> pointIdxVec;
		float source_num=0;
		float target_num=0;
		if (octree.voxelSearch ((*(vec.begin()+i)), pointIdxVec))
		{	
			int r=rand()%255;
			int g=rand()%255;
			int b=rand()%255;
			for(auto it1=pointIdxVec.begin();it1!=pointIdxVec.end();it1++)
			{
				pcl::PointXYZRGB pointrgbtemp;
				pointrgbtemp.x=((*cloud_fuse)[(*it1)]).x;
				pointrgbtemp.y=((*cloud_fuse)[(*it1)]).y;
				pointrgbtemp.z=((*cloud_fuse)[(*it1)]).z;
				pointrgbtemp.r=r;
				pointrgbtemp.g=g;
				pointrgbtemp.b=b;
				(*cloudecorlor).push_back(pointrgbtemp);				
				if(((*cloud_fuse)[(*it1)].intensity)==0)
				{
					source_num++;
				}else if (((*cloud_fuse)[(*it1)].intensity)==1)
				{
					target_num++;
				}	
			}
		}	
		allinvec=allinvec+source_num+target_num;	
		if(source_num>=threash_num && target_num>=threash_num)
		{
			float min=source_num<target_num?source_num:target_num;
			float max=source_num>target_num?source_num:target_num;
			similar_num=similar_num+(source_num+target_num)*(min/max);
		}
	}
	float score=similar_num/allinvec;
	return score;	
}

void computer_transform(std::vector<std::vector<Eigen::Matrix4f>> &transformation_vecter,int index11,int index12,int index21,int index22,std::vector<facenode> &face_vecter1,std::vector<facenode> &face_vecter2, int type)
{
	Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity ();
	Eigen::Vector3f n1=Eigen::Vector3f(face_vecter1[index11].average_normal_x,face_vecter1[index11].average_normal_y,face_vecter1[index11].average_normal_z);
	Eigen::Vector3f m1=Eigen::Vector3f(face_vecter1[index12].average_normal_x,face_vecter1[index12].average_normal_y,face_vecter1[index12].average_normal_z);
	Eigen::Vector3f n2=Eigen::Vector3f(face_vecter2[index21].average_normal_x,face_vecter2[index21].average_normal_y,face_vecter2[index21].average_normal_z);
	Eigen::Vector3f m2=Eigen::Vector3f(face_vecter2[index22].average_normal_x,face_vecter2[index22].average_normal_y,face_vecter2[index22].average_normal_z); 
	Eigen::Vector3f r1=n2.cross(n1);
	r1.normalize(); 
	Eigen::Matrix3f r1x = Eigen::Matrix3f::Identity();
	r1x(0,0)=0;
	r1x(0,1)=-r1[2];
	r1x(0,2)=r1[1];
	r1x(1,0)=r1[2];
	r1x(1,1)=0;
	r1x(1,2)=-r1[0];
	r1x(2,0)=-r1[1];
	r1x(2,1)=r1[0];
	r1x(2,2)=0;	
	Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
	Eigen::Matrix3f R1 = Eigen::Matrix3f::Identity();
	float n2dn1=n2.dot(n1);
	Eigen::Vector3f r1cn2=r1.cross(n2);
	float r1cn2dn1=r1cn2.dot(n1);
	float cos_theta1=n2dn1;
	float sin_theta1=r1cn2dn1;
	Eigen::Matrix3f rrt1 = r1*(r1.transpose());
	R1=cos_theta1*I+(1-cos_theta1)*rrt1+sin_theta1*r1x;
	
	m2=R1*m2;

	Eigen::Matrix3f R2 = Eigen::Matrix3f::Identity();
	Eigen::Vector3f r2=n1;
	Eigen::Matrix3f r2x = Eigen::Matrix3f::Identity();
	r2x(0,0)=0;
	r2x(0,1)=-r2[2];
	r2x(0,2)=r2[1];
	r2x(1,0)=r2[2];
	r2x(1,1)=0;
	r2x(1,2)=-r2[0];
	r2x(2,0)=-r2[1];
	r2x(2,1)=r2[0];
	r2x(2,2)=0;	
	Eigen::Matrix3f rrt2 = r2*(r2.transpose());
	float m2dm1=m2.dot(m1);
	float m2dr2=m2.dot(r2);
	float m1dr2=m1.dot(r2);
	Eigen::Vector3f r2cm2=r2.cross(m2);
	float r2cm2dm1=r2cm2.dot(m1);
	float cos_theta2=(m2dm1-(m2dr2*m1dr2))/(1-(m2dr2*m1dr2));
	float sin_theta2=(r2cm2dm1)/(1-(m2dr2*m1dr2));
	R2=cos_theta2*I+(1-cos_theta2)*rrt2+sin_theta2*r2x;

	Eigen::Matrix3f rotMatrix=Eigen::Matrix3f::Identity();
	rotMatrix=R2*R1;
	transformation_matrix(0,0)=rotMatrix(0,0);
	transformation_matrix(0,1)=rotMatrix(0,1);
	transformation_matrix(0,2)=rotMatrix(0,2);
	transformation_matrix(1,0)=rotMatrix(1,0);
	transformation_matrix(1,1)=rotMatrix(1,1);
	transformation_matrix(1,2)=rotMatrix(1,2);
	transformation_matrix(2,0)=rotMatrix(2,0);
	transformation_matrix(2,1)=rotMatrix(2,1);
	transformation_matrix(2,2)=rotMatrix(2,2);

	std::vector<face_three> face_three_vecter1;
	Eigen::Vector3f n1cm1=n1.cross(m1);
	n1cm1.normalize();
	float chose_threashold=third_plane_threshold;
	int face1_index=0;
	for(auto it1=face_vecter1.begin();it1!=face_vecter1.end();it1++)
	{
		if((face1_index!=index11) && (face1_index!=index12))
		{
			Eigen::Vector3f nthree=Eigen::Vector3f((*it1).average_normal_x,(*it1).average_normal_y,(*it1).average_normal_z);
			float n1cm1dnthree=n1cm1.dot(nthree);
			if(fabs(n1cm1dnthree)>chose_threashold)
			{
				face_three face_three_temp;
				face_three_temp.index1=index11;
				face_three_temp.index2=index12;
				face_three_temp.index3=face1_index;
				face_three_vecter1.push_back(face_three_temp);
			}
		}
		face1_index++;
	}

	////////////////
	Eigen::Vector3f n2cm2=n2.cross(m2);
	n2cm2.normalize();
	////////////////
	bool getthree=false;
	if(face_three_vecter1.size()>0)
	{
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normal(new pcl::PointCloud< pcl::PointNormal>);
		for(auto it1=face_vecter2.begin();it1!=face_vecter2.end();it1++)
		{
			pcl::PointNormal normal_temp;
			normal_temp.x=(*it1).average_centry_x;
			normal_temp.y=(*it1).average_centry_y;
			normal_temp.z=(*it1).average_centry_z;
			normal_temp.normal_x=(*it1).average_normal_x;
			normal_temp.normal_y=(*it1).average_normal_y;
			normal_temp.normal_z=(*it1).average_normal_z;
			(*cloud_normal).push_back(normal_temp);
		}
		pcl::transformPointCloudWithNormals(*cloud_normal, *cloud_normal, transformation_matrix);
		float angelthreash=third_plane_normal_threshold;
		for(auto it1=face_three_vecter1.begin();it1!=face_three_vecter1.end();it1++)
		{
			int face2_index=0;
			for(const auto& point: *cloud_normal)
			{
				if((face2_index!=index21) && (face2_index!=index22))
				{
					float angel_of_three=compute_normal_angel(face_vecter1[(*it1).index3].average_normal_x,face_vecter1[(*it1).index3].average_normal_y,face_vecter1[(*it1).index3].average_normal_z,point.normal_x,point.normal_y,point.normal_z);
					if(angel_of_three<angelthreash && fabs(n2cm2.dot(Eigen::Vector3f(point.normal_x,point.normal_y,point.normal_z))) > chose_threashold) // fabs(n1m1k1.dot(n2m2k2)) > 0.5 && fabs(n2cm2.dot(Eigen::Vector3f(point.normal_x,point.normal_y,point.normal_z))) > chose_threashold
					{
						getthree=true;
						Eigen::Vector3f k1=Eigen::Vector3f(face_vecter1[(*it1).index3].average_normal_x,face_vecter1[(*it1).index3].average_normal_y,face_vecter1[(*it1).index3].average_normal_z);
						Eigen::Vector3f k2=Eigen::Vector3f(point.normal_x,point.normal_y,point.normal_z);
						Eigen::Vector3f c11=Eigen::Vector3f(face_vecter1[index11].average_centry_x,face_vecter1[index11].average_centry_y,face_vecter1[index11].average_centry_z);
						Eigen::Vector3f c12=Eigen::Vector3f(face_vecter1[index12].average_centry_x,face_vecter1[index12].average_centry_y,face_vecter1[index12].average_centry_z);
						Eigen::Vector3f c13=Eigen::Vector3f(face_vecter1[(*it1).index3].average_centry_x,face_vecter1[(*it1).index3].average_centry_y,face_vecter1[(*it1).index3].average_centry_z);
						Eigen::Vector3f c21=Eigen::Vector3f(face_vecter2[index21].average_centry_x,face_vecter2[index21].average_centry_y,face_vecter2[index21].average_centry_z);
						Eigen::Vector3f c22=Eigen::Vector3f(face_vecter2[index22].average_centry_x,face_vecter2[index22].average_centry_y,face_vecter2[index22].average_centry_z);
						Eigen::Vector3f c23=Eigen::Vector3f(point.x,point.y,point.z);
						float d11=c11.dot(n1);
						float d12=c12.dot(m1);
						float d13=c13.dot(k1);
						float d21=c21.dot(n2);
						float d22=c22.dot(m2);
						float d23=c23.dot(k2);
						Eigen::Vector3f D=Eigen::Vector3f(d11-d21,d12-d22,d13-d23);
						Eigen::Matrix3f A = Eigen::Matrix3f::Identity();
						A(0,0)=n1[0];
						A(0,1)=n1[1];
						A(0,2)=n1[2];
						A(1,0)=m1[0];
						A(1,1)=m1[1];
						A(1,2)=m1[2];
						A(2,0)=k1[0];
						A(2,1)=k1[1];
						A(2,2)=k1[2];
						Eigen::Matrix3f AT=A.transpose();
						Eigen::Vector3f T=((AT*A).inverse())*AT*D;
					    transformation_matrix(0,3)=T[0];
		                transformation_matrix(1,3)=T[1];
		                transformation_matrix(2,3)=T[2];						
						Eigen::Matrix4f transformation_matrix_temp= Eigen::Matrix4f::Identity ();
						transformation_matrix_temp=transformation_matrix;
						transformation_vecter[type].push_back(transformation_matrix_temp);												
					}
				}
				face2_index++;
			}
		}
	}
	if(getthree==false)
	{
		float sourcefacecenterx=(face_vecter1[index11].average_centry_x*face_vecter1[index11].face_point_size+face_vecter1[index12].average_centry_x*face_vecter1[index12].face_point_size)/(face_vecter1[index11].face_point_size+face_vecter1[index12].face_point_size);
		float sourcefacecentery=(face_vecter1[index11].average_centry_y*face_vecter1[index11].face_point_size+face_vecter1[index12].average_centry_y*face_vecter1[index12].face_point_size)/(face_vecter1[index11].face_point_size+face_vecter1[index12].face_point_size);
		float sourcefacecenterz=(face_vecter1[index11].average_centry_z*face_vecter1[index11].face_point_size+face_vecter1[index12].average_centry_z*face_vecter1[index12].face_point_size)/(face_vecter1[index11].face_point_size+face_vecter1[index12].face_point_size);
		float targetfacecenterx=(face_vecter2[index21].average_centry_x*face_vecter2[index21].face_point_size+face_vecter2[index22].average_centry_x*face_vecter2[index22].face_point_size)/(face_vecter2[index21].face_point_size+face_vecter2[index22].face_point_size);
		float targetfacecentery=(face_vecter2[index21].average_centry_y*face_vecter2[index21].face_point_size+face_vecter2[index22].average_centry_y*face_vecter2[index22].face_point_size)/(face_vecter2[index21].face_point_size+face_vecter2[index22].face_point_size);
		float targetfacecenterz=(face_vecter2[index21].average_centry_z*face_vecter2[index21].face_point_size+face_vecter2[index22].average_centry_z*face_vecter2[index22].face_point_size)/(face_vecter2[index21].face_point_size+face_vecter2[index22].face_point_size);
		Eigen::Vector3f sourcefacecenter=Eigen::Vector3f(sourcefacecenterx,sourcefacecentery,sourcefacecenterz);
		Eigen::Vector3f targetfacecenter=Eigen::Vector3f(targetfacecenterx,targetfacecentery,targetfacecenterz);
		targetfacecenter=rotMatrix*targetfacecenter;			
		transformation_matrix(0,3)=sourcefacecenter[0]-targetfacecenter[0];
		transformation_matrix(1,3)=sourcefacecenter[1]-targetfacecenter[1];
		transformation_matrix(2,3)=sourcefacecenter[2]-targetfacecenter[2];
		Eigen::Matrix4f transformation_matrix_temp= Eigen::Matrix4f::Identity();
		transformation_matrix_temp=transformation_matrix;
		transformation_vecter[type].push_back(transformation_matrix_temp);		
	}
}

void range_cluster(std::vector<std::vector<transform_q_t> > &transform_q_t_vector_cluster)
{
	for(int i=0;i<int(transform_q_t_vector_cluster.size());i++)
	{
		for(int j=0;j<int(transform_q_t_vector_cluster.size());j++)
		{
			if(j>i)
			{
				if(transform_q_t_vector_cluster[i].size()<transform_q_t_vector_cluster[j].size())
				{
					std::vector<transform_q_t> transform_q_t_vector_temp;
					transform_q_t_vector_temp=transform_q_t_vector_cluster[i];
					transform_q_t_vector_cluster[i]=transform_q_t_vector_cluster[j];
					transform_q_t_vector_cluster[j]=transform_q_t_vector_temp;
				}
			}
		}		
	}
}

void transform_cluster(std::vector<transform_q_t> &transform_q_t_vecter, std::vector<transform_q_t> &fine_transform_q_t_vecter, int cluster_num)
{
	int transform_q_t_vecter_size=transform_q_t_vecter.size();
	if(transform_q_t_vecter_size<=cluster_number_threshold)
	{
		if(transform_q_t_vecter_size==0)
		{
			transform_q_t transform_q_t_average_temp;
			transform_q_t_average_temp.qw = 1;
			transform_q_t_average_temp.qx = 0;
			transform_q_t_average_temp.qy = 0;
			transform_q_t_average_temp.qz = 0;
			transform_q_t_average_temp.tx = 0;
			transform_q_t_average_temp.ty = 0;
			transform_q_t_average_temp.tz = 0;
			transform_q_t_average_temp.is_allocate = true;
			fine_transform_q_t_vecter.push_back(transform_q_t_average_temp);			
		}else
		{
			for (auto it1 = transform_q_t_vecter.begin(); it1 != transform_q_t_vecter.end(); it1++)
			{	
				fine_transform_q_t_vecter.push_back((*it1));
			}			
		}	
	}else
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr t_cloud (new pcl::PointCloud<pcl::PointXYZ>);
		for (auto it1 = transform_q_t_vecter.begin(); it1 != transform_q_t_vecter.end(); it1++)
		{	
			pcl::PointXYZ pointxyz_temp;
			pointxyz_temp.x=(*it1).tx;
			pointxyz_temp.y=(*it1).ty;
			pointxyz_temp.z=(*it1).tz;
			(*t_cloud).push_back(pointxyz_temp);
		}
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud (t_cloud);

		float distantq_threashold = cluster_angel_threshold;
		float distantt_threashold = cluster_distance_threshold;
		std::vector<std::vector<transform_q_t> > transform_q_t_vector_cluster;
		int it1_index=0;
		for (auto it1 = transform_q_t_vecter.begin(); it1 != transform_q_t_vecter.end(); it1++)
		{
			if (it1 != (transform_q_t_vecter.end() - 1))
			{
				if ((*it1).is_allocate == false)
				{
					std::vector<transform_q_t> transform_q_t_vector_temp;
					std::vector<int> pointIdxRadiusSearch;
					std::vector<float> pointRadiusSquaredDistance;  
					if ( kdtree.radiusSearch ((*t_cloud)[it1_index], distantt_threashold, pointIdxRadiusSearch,pointRadiusSquaredDistance) > 0 )
					{
						for (auto it2 = pointIdxRadiusSearch.begin(); it2 != pointIdxRadiusSearch.end(); it2++)
						{
							Eigen::Quaternionf Q1;
							Q1.w()=(*it1).qw;
							Q1.x()=(*it1).qx;
							Q1.y()=(*it1).qy;
							Q1.z()=(*it1).qz;
							Eigen::Quaternionf Q2;
							Q2.w()=transform_q_t_vecter[(*it2)].qw;
							Q2.x()=transform_q_t_vecter[(*it2)].qx;
							Q2.y()=transform_q_t_vecter[(*it2)].qy;
							Q2.z()=transform_q_t_vecter[(*it2)].qz;	
							Eigen::Vector3f p1 = Eigen::Vector3f(1,0,0);
							Eigen::Vector3f p2 = Eigen::Vector3f(1,0,0);
							p1=Q1*p1;
							p2=Q2*p2;
							float distanceq=compute_normal_angel(p1[0],p1[1],p1[2],p2[0],p2[1],p2[2]);
							if(distanceq<distantq_threashold)
							{
								transform_q_t_vecter[(*it2)].is_allocate = true;
								transform_q_t_vector_temp.push_back(transform_q_t_vecter[(*it2)]);
							}
						}
					}			
					transform_q_t_vector_cluster.push_back(transform_q_t_vector_temp);
				}
			}
			it1_index++;
		}
		range_cluster(transform_q_t_vector_cluster);
		int clusternum=(*transform_q_t_vector_cluster.begin()).size();
		bool stop=false;

		for (auto it1 = transform_q_t_vector_cluster.begin(); it1 != transform_q_t_vector_cluster.end(); it1++)
		{
			if(stop==false)
			{
				if(int((*it1).size())>=clusternum)
				{
					float averagetx = 0;
					float averagety = 0;
					float averagetz = 0;
					for (auto it2 = (*it1).begin(); it2 != (*it1).end(); it2++)
					{
						averagetx = averagetx + (*it2).tx;
						averagety = averagety + (*it2).ty;
						averagetz = averagetz + (*it2).tz;
					}
					averagetx = averagetx / (*it1).size();
					averagety = averagety / (*it1).size();
					averagetz = averagetz / (*it1).size();

					Eigen::Vector3f rotationvector1;
					Eigen::Vector3f rotationvector2;
					average_normal(rotationvector1,rotationvector2,(*it1));
					Eigen::Vector3f nt1=Eigen::Vector3f(rotationvector1[0],rotationvector1[1],rotationvector1[2]);
					Eigen::Vector3f nt2=Eigen::Vector3f(rotationvector2[0],rotationvector2[1],rotationvector2[2]);
					Eigen::Vector3f ns1=Eigen::Vector3f(1,0,0); 
					Eigen::Vector3f ns2=Eigen::Vector3f(0,1,0);

					Eigen::Vector3f r1=ns1.cross(nt1);
					r1.normalize(); 
					Eigen::Matrix3f r1x = Eigen::Matrix3f::Identity();
					r1x(0,0)=0;
					r1x(0,1)=-r1[2];
					r1x(0,2)=r1[1];
					r1x(1,0)=r1[2];
					r1x(1,1)=0;
					r1x(1,2)=-r1[0];
					r1x(2,0)=-r1[1];
					r1x(2,1)=r1[0];
					r1x(2,2)=0;	
					Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
					Eigen::Matrix3f R1 = Eigen::Matrix3f::Identity();
					float cos_theta1=nt1.dot(ns1);
					float sin_theta1=nt1.dot(r1.cross(ns1));
					Eigen::Matrix3f rrt1 = r1*(r1.transpose());
					R1=cos_theta1*I+(1-cos_theta1)*rrt1+sin_theta1*r1x;

					ns2=R1*ns2;
					Eigen::Vector3f r2=nt1;
					Eigen::Matrix3f r2x = Eigen::Matrix3f::Identity();
					r2x(0,0)=0;
					r2x(0,1)=-r2[2];
					r2x(0,2)=r2[1];
					r2x(1,0)=r2[2];
					r2x(1,1)=0;
					r2x(1,2)=-r2[0];
					r2x(2,0)=-r2[1];
					r2x(2,1)=r2[0];
					r2x(2,2)=0;	
					float ns2dnt2=ns2.dot(nt2);
					float ns2dr2=ns2.dot(r2);
					float nt2dr2=nt2.dot(r2);
					Eigen::Vector3f r2cns2=r2.cross(ns2);
					float r2cns2dnt2=r2cns2.dot(nt2);				
					float cos_theta2=(ns2dnt2-(ns2dr2*nt2dr2))/(1-(ns2dr2*nt2dr2));
					float sin_theta2=(r2cns2dnt2)/(1-(ns2dr2*nt2dr2));
					Eigen::Matrix3f rrt2 = r2*(r2.transpose());
					Eigen::Matrix3f R2 = Eigen::Matrix3f::Identity();
					R2=cos_theta2*I+(1-cos_theta2)*rrt2+sin_theta2*r2x;

					Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
					R = R2*R1;
					Eigen::Quaternionf q(R);
					transform_q_t transform_q_t_average_temp;
					transform_q_t_average_temp.qw = q.w();
					transform_q_t_average_temp.qx = q.x();
					transform_q_t_average_temp.qy = q.y();
					transform_q_t_average_temp.qz = q.z();
					transform_q_t_average_temp.tx = averagetx;
					transform_q_t_average_temp.ty = averagety;
					transform_q_t_average_temp.tz = averagetz;
					transform_q_t_average_temp.is_allocate = true;
					fine_transform_q_t_vecter.push_back(transform_q_t_average_temp);
					if(fine_transform_q_t_vecter.size()>cluster_num)
					{
						break;
					}
				}
				else
				{
					if(fine_transform_q_t_vecter.size()<(cluster_num/2.0))
					{
						stop=false;
						clusternum--;
						if(clusternum<2)
						{
							break;
						}
					}else
					{
						stop=true;
					}
				}
			}
		}
	}
}

void score_range(std::vector<transform_score> &cluster_transformation_vecter)
{
	for (auto it1 = cluster_transformation_vecter.begin(); it1 != cluster_transformation_vecter.end(); it1++)
	{
		if (it1 != (cluster_transformation_vecter.end() - 1))
		{
			for (auto it2 = it1 + 1; it2 != cluster_transformation_vecter.end(); it2++)
			{
				if ((*it1).score < (*it2).score)
				{
					transform_score transform_score_temp;
					transform_score_temp = (*it1);
					(*it1) = (*it2);
					(*it2) = transform_score_temp;
				}
			}
		}
	}
}

void weight_normal(Eigen::Vector3f &rotaionvector1,Eigen::Vector3f &rotaionvector2,std::vector<high_score> &highest_score, float sum_score)
{
	float sumnx1=0;
	float sumny1=0;
	float sumnz1=0;
	float sumnx2=0;
	float sumny2=0;
	float sumnz2=0;	
	for(auto it1=highest_score.begin();it1!=highest_score.end();it1++)
	{
		Eigen::Vector3f n1=Eigen::Vector3f(1,0,0);
		Eigen::Vector3f n2=Eigen::Vector3f(0,1,0);
		Eigen::Quaternionf Q;
		Q.w()=(*it1).qt.qw;
		Q.x()=(*it1).qt.qx;
		Q.y()=(*it1).qt.qy;
		Q.z()=(*it1).qt.qz;
		n1=Q*n1;
		n2=Q*n2;
		sumnx1=sumnx1+n1[0]*((*it1).score/sum_score);
		sumny1=sumny1+n1[1]*((*it1).score/sum_score);
		sumnz1=sumnz1+n1[2]*((*it1).score/sum_score);
		sumnx2=sumnx2+n2[0]*((*it1).score/sum_score);
		sumny2=sumny2+n2[1]*((*it1).score/sum_score);
		sumnz2=sumnz2+n2[2]*((*it1).score/sum_score);		
	}
	Eigen::Vector3f averagen1=Eigen::Vector3f(sumnx1,sumny1,sumnz1);
	Eigen::Vector3f averagen2=Eigen::Vector3f(sumnx2,sumny2,sumnz2);
	averagen1.normalize();
	averagen2.normalize();
	rotaionvector1[0]=averagen1[0];
	rotaionvector1[1]=averagen1[1];
	rotaionvector1[2]=averagen1[2];
	rotaionvector2[0]=averagen2[0];
	rotaionvector2[1]=averagen2[1];
	rotaionvector2[2]=averagen2[2];	
}

void fuse_answer(Eigen::Matrix4f &best_transformation_matrix, std::vector<high_score> &highest_score, float sum_score)
{
	float averagetx = 0;
	float averagety = 0;
	float averagetz = 0;
	for (auto it1 = (highest_score).begin(); it1 != (highest_score).end(); it1++)
	{
		averagetx = averagetx + (*it1).qt.tx*((*it1).score/sum_score);
		averagety = averagety + (*it1).qt.ty*((*it1).score/sum_score);
		averagetz = averagetz + (*it1).qt.tz*((*it1).score/sum_score);
	}

	Eigen::Vector3f rotationvector1;
	Eigen::Vector3f rotationvector2;
	weight_normal(rotationvector1,rotationvector2,highest_score,sum_score);
	Eigen::Vector3f nt1=Eigen::Vector3f(rotationvector1[0],rotationvector1[1],rotationvector1[2]);
	Eigen::Vector3f nt2=Eigen::Vector3f(rotationvector2[0],rotationvector2[1],rotationvector2[2]);
	Eigen::Vector3f ns1=Eigen::Vector3f(1,0,0); 
	Eigen::Vector3f ns2=Eigen::Vector3f(0,1,0);

	Eigen::Vector3f r1=ns1.cross(nt1);
	r1.normalize(); 
	Eigen::Matrix3f r1x = Eigen::Matrix3f::Identity();
	r1x(0,0)=0;
	r1x(0,1)=-r1[2];
	r1x(0,2)=r1[1];
	r1x(1,0)=r1[2];
	r1x(1,1)=0;
	r1x(1,2)=-r1[0];
	r1x(2,0)=-r1[1];
	r1x(2,1)=r1[0];
	r1x(2,2)=0;	
	Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
	Eigen::Matrix3f R1 = Eigen::Matrix3f::Identity();
	float cos_theta1=nt1.dot(ns1);
	float sin_theta1=nt1.dot(r1.cross(ns1));
	Eigen::Matrix3f rrt1 = r1*(r1.transpose());
	R1=cos_theta1*I+(1-cos_theta1)*rrt1+sin_theta1*r1x;

	ns2=R1*ns2;
	Eigen::Vector3f r2=nt1;
	Eigen::Matrix3f r2x = Eigen::Matrix3f::Identity();
	r2x(0,0)=0;
	r2x(0,1)=-r2[2];
	r2x(0,2)=r2[1];
	r2x(1,0)=r2[2];
	r2x(1,1)=0;
	r2x(1,2)=-r2[0];
	r2x(2,0)=-r2[1];
	r2x(2,1)=r2[0];
	r2x(2,2)=0;	
	float ns2dnt2=ns2.dot(nt2);
	float ns2dr2=ns2.dot(r2);
	float nt2dr2=nt2.dot(r2);
	Eigen::Vector3f r2cns2=r2.cross(ns2);
	float r2cns2dnt2=r2cns2.dot(nt2);				
	float cos_theta2=(ns2dnt2-(ns2dr2*nt2dr2))/(1-(ns2dr2*nt2dr2));
	float sin_theta2=(r2cns2dnt2)/(1-(ns2dr2*nt2dr2));
	Eigen::Matrix3f rrt2 = r2*(r2.transpose());
	Eigen::Matrix3f R2 = Eigen::Matrix3f::Identity();
	R2=cos_theta2*I+(1-cos_theta2)*rrt2+sin_theta2*r2x;

	Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
	R = R2*R1;

	(best_transformation_matrix)(0, 0) = R(0, 0);
	(best_transformation_matrix)(0, 1) = R(0, 1);
	(best_transformation_matrix)(0, 2) = R(0, 2);
	(best_transformation_matrix)(1, 0) = R(1, 0);
	(best_transformation_matrix)(1, 1) = R(1, 1);
	(best_transformation_matrix)(1, 2) = R(1, 2);
	(best_transformation_matrix)(2, 0) = R(2, 0);
	(best_transformation_matrix)(2, 1) = R(2, 1);
	(best_transformation_matrix)(2, 2) = R(2, 2);
	(best_transformation_matrix)(0, 3) = averagetx;
	(best_transformation_matrix)(1, 3) = averagety;
	(best_transformation_matrix)(2, 3) = averagetz;
}

void computer_transform_guess(pcl::PointCloud<pcl::PointXYZ>::Ptr source,pcl::PointCloud<pcl::PointXYZ>::Ptr target,Eigen::Matrix4f &best_transformation_matrix)
{
	std::vector<int> indices1;
	std::vector<int> indices2;
	pcl::removeNaNFromPointCloud(*source,*source,indices1);
	pcl::removeNaNFromPointCloud(*target,*target,indices2);

	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_source;
	voxel_grid_source.setLeafSize(LeafSize, LeafSize, LeafSize);
	voxel_grid_source.setInputCloud(source);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
	voxel_grid_source.filter(*cloud_src);

	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_target;
	voxel_grid_target.setLeafSize(LeafSize, LeafSize, LeafSize);
	voxel_grid_target.setInputCloud(target);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tag(new pcl::PointCloud<pcl::PointXYZ>);
	voxel_grid_target.filter(*cloud_tag);


	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sub1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sub2(new pcl::PointCloud<pcl::PointXYZ>);	
	std::vector<double> theta_vector1;
	std::vector<double> theta_vector2;
	
	
	std::vector<facenode> face_vecter1;
	std::vector<facenode> face_vecter2;
	std::vector<double> new_theta_vector1;
	std::vector<double> new_theta_vector2;
	face_extrate(cloud_src,face_vecter1,cloud_sub1,new_theta_vector1);
	face_extrate(cloud_tag,face_vecter2,cloud_sub2,new_theta_vector2);
	
	std::vector<int> type_index1;
	std::vector<int> type_index2;
	std::vector<face_base> base_vecter1;
	select_base(base_vecter1, face_vecter1, type_index1, new_theta_vector1);

	std::vector<face_base> base_vecter2;
	select_base(base_vecter2, face_vecter2, type_index2, new_theta_vector2);
	std::vector<std::vector<Eigen::Matrix4f>> transformation_vecter(3,std::vector<Eigen::Matrix4f>(0));
    
	float angelthreash = included_angle_same_threshold;
	
	int i1 = 0; //
	for (auto it1 = base_vecter1.begin(); it1 != base_vecter1.end(); it1++)
	{
		int i2 = 0; //
		for (auto it2 = base_vecter2.begin(); it2 != base_vecter2.end(); it2++)
		{
			if ((fabs((*it1).angel - (*it2).angel)) < angelthreash && type_index1[i1]==type_index2[i2]) //  && matching_rate[(*it1).index1][(*it2).index1] >= 0.05 && matching_rate[(*it1).index2][(*it2).index2] >= 0.05
			{
					computer_transform(transformation_vecter, (*it1).index1, (*it1).index2, (*it2).index1, (*it2).index2, face_vecter1, face_vecter2, type_index1[i1]);
			}
			i2++; //
		}
		i1++; //
	}
	int transformation_num = transformation_vecter[0].size()+transformation_vecter[1].size()+transformation_vecter[2].size();
	
	float score_sum = 0;
	float score1_sum = 0;
	float score2_sum = 0;
	std::vector<std::vector<transform_score>> cluster_transformation_vecter(3,std::vector<transform_score>(0));
	int analyse_max = fine_verify_number;
	int analyse_sum = 0;

	for(auto i = 0; i < transformation_vecter.size(); i++){
		std::vector<transform_q_t> transform_q_t_vecter;
		for (auto it1 = transformation_vecter[i].begin(); it1 != transformation_vecter[i].end(); it1++)
		{
			Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
			R(0, 0) = (*it1)(0, 0);
			R(0, 1) = (*it1)(0, 1);
			R(0, 2) = (*it1)(0, 2);
			R(1, 0) = (*it1)(1, 0);
			R(1, 1) = (*it1)(1, 1);
			R(1, 2) = (*it1)(1, 2);
			R(2, 0) = (*it1)(2, 0);
			R(2, 1) = (*it1)(2, 1);
			R(2, 2) = (*it1)(2, 2);
			Eigen::Quaternionf quaterniond_temp(R);
			transform_q_t transform_q_t_temp;
			transform_q_t_temp.qw = quaterniond_temp.w();
			transform_q_t_temp.qx = quaterniond_temp.x();
			transform_q_t_temp.qy = quaterniond_temp.y();
			transform_q_t_temp.qz = quaterniond_temp.z();
			transform_q_t_temp.tx = (*it1)(0, 3);
			transform_q_t_temp.ty = (*it1)(1, 3);
			transform_q_t_temp.tz = (*it1)(2, 3);
			transform_q_t_temp.is_allocate = false;
			transform_q_t_vecter.push_back(transform_q_t_temp);
		}

		std::vector<transform_q_t> fine_transform_q_t_vecter;
		int cluster_num = seclct_cluster_number*transformation_vecter[i].size()/transformation_num;
		transform_cluster(transform_q_t_vecter, fine_transform_q_t_vecter, cluster_num);

		for (auto it1 = fine_transform_q_t_vecter.begin(); it1 != fine_transform_q_t_vecter.end(); it1++)
		{
			Eigen::Quaternionf quaterniond_temp;
			quaterniond_temp.w() = (*it1).qw;
			quaterniond_temp.x() = (*it1).qx;
			quaterniond_temp.y() = (*it1).qy;
			quaterniond_temp.z() = (*it1).qz;
			Eigen::Matrix3f R = quaterniond_temp.toRotationMatrix();
			transform_score transform_score_temp;
			(transform_score_temp.transformation_matrix) = Eigen::Matrix4f::Identity();
			(transform_score_temp.transformation_matrix)(0, 0) = R(0, 0);
			(transform_score_temp.transformation_matrix)(0, 1) = R(0, 1);
			(transform_score_temp.transformation_matrix)(0, 2) = R(0, 2);
			(transform_score_temp.transformation_matrix)(1, 0) = R(1, 0);
			(transform_score_temp.transformation_matrix)(1, 1) = R(1, 1);
			(transform_score_temp.transformation_matrix)(1, 2) = R(1, 2);
			(transform_score_temp.transformation_matrix)(2, 0) = R(2, 0);
			(transform_score_temp.transformation_matrix)(2, 1) = R(2, 1);
			(transform_score_temp.transformation_matrix)(2, 2) = R(2, 2);
			(transform_score_temp.transformation_matrix)(0, 3) = (*it1).tx;
			(transform_score_temp.transformation_matrix)(1, 3) = (*it1).ty;
			(transform_score_temp.transformation_matrix)(2, 3) = (*it1).tz;
			(transform_score_temp.score) = quick_verify((transform_score_temp.transformation_matrix), face_vecter1, face_vecter2);
			cluster_transformation_vecter[i].push_back(transform_score_temp);
		}

		score_range(cluster_transformation_vecter[i]);

		analyse_sum = 0;

		
		for (auto it1 = cluster_transformation_vecter[i].begin(); it1 != cluster_transformation_vecter[i].end(); it1++)
		{
			if (analyse_sum < analyse_max)
			{
				analyse_sum++;
				Eigen::Matrix4f transformation_matrix_temp = Eigen::Matrix4f::Identity();
				transformation_matrix_temp(0, 0) = ((*it1).transformation_matrix)(0, 0);
				transformation_matrix_temp(0, 1) = ((*it1).transformation_matrix)(0, 1);
				transformation_matrix_temp(0, 2) = ((*it1).transformation_matrix)(0, 2);
				transformation_matrix_temp(0, 3) = ((*it1).transformation_matrix)(0, 3);
				transformation_matrix_temp(1, 0) = ((*it1).transformation_matrix)(1, 0);
				transformation_matrix_temp(1, 1) = ((*it1).transformation_matrix)(1, 1);
				transformation_matrix_temp(1, 2) = ((*it1).transformation_matrix)(1, 2);
				transformation_matrix_temp(1, 3) = ((*it1).transformation_matrix)(1, 3);
				transformation_matrix_temp(2, 0) = ((*it1).transformation_matrix)(2, 0);
				transformation_matrix_temp(2, 1) = ((*it1).transformation_matrix)(2, 1);
				transformation_matrix_temp(2, 2) = ((*it1).transformation_matrix)(2, 2);
				transformation_matrix_temp(2, 3) = ((*it1).transformation_matrix)(2, 3);
				pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZI>);
				for (auto &point : *cloud_sub1)
				{
					pcl::PointXYZI pointxyzi_temp;
					pointxyzi_temp.x = point.x;
					pointxyzi_temp.y = point.y;
					pointxyzi_temp.z = point.z;
					pointxyzi_temp.intensity = 0;
					(*cloud_source).push_back(pointxyzi_temp);
				}
				pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZI>);
				for (auto &point : *cloud_sub2)
				{				
					pcl::PointXYZI pointxyzi_temp;
					pointxyzi_temp.x = point.x;
					pointxyzi_temp.y = point.y;
					pointxyzi_temp.z = point.z;
					pointxyzi_temp.intensity = 1;
					(*cloud_target).push_back(pointxyzi_temp);
				}

				it1->score2 = fine_verify(transformation_matrix_temp, cloud_source, cloud_target);
				score2_sum += (*it1).score2; // fusion score1+2
				score1_sum += (*it1).score; // fusion score1+2
			}
			else
				break;
		}	
	}
	std::vector<high_score> tmp_highest_3score;
	float best_best_score = 0;
	for(auto i = 0; i < transformation_vecter.size(); i++){
		// fusion score1+2
		analyse_sum = 0;
		float best_score = 0;
		Eigen::Matrix4f tmp_best_transformation_matrix = Eigen::Matrix4f::Identity();
		for (auto it1 = cluster_transformation_vecter[i].begin(); it1 != cluster_transformation_vecter[i].end(); it1++)
		{
			if (analyse_sum < analyse_max)
			{
				analyse_sum++;
				float score = (*it1).score/score1_sum + (*it1).score2/score2_sum;
				if (score > best_score)
				{
					best_score = score;
					tmp_best_transformation_matrix = (*it1).transformation_matrix;
				}
			}
		}
		if(best_best_score < best_score)
			best_best_score = best_score;

		// transfer to qt
		Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
		R(0, 0) = (tmp_best_transformation_matrix)(0, 0);
		R(0, 1) = (tmp_best_transformation_matrix)(0, 1);
		R(0, 2) = (tmp_best_transformation_matrix)(0, 2);
		R(1, 0) = (tmp_best_transformation_matrix)(1, 0);
		R(1, 1) = (tmp_best_transformation_matrix)(1, 1);
		R(1, 2) = (tmp_best_transformation_matrix)(1, 2);
		R(2, 0) = (tmp_best_transformation_matrix)(2, 0);
		R(2, 1) = (tmp_best_transformation_matrix)(2, 1);
		R(2, 2) = (tmp_best_transformation_matrix)(2, 2);
		Eigen::Quaternionf quaterniond_temp(R);
		transform_q_t transform_q_t_temp;
		transform_q_t_temp.qw = quaterniond_temp.w();
		transform_q_t_temp.qx = quaterniond_temp.x();
		transform_q_t_temp.qy = quaterniond_temp.y();
		transform_q_t_temp.qz = quaterniond_temp.z();
		transform_q_t_temp.tx = (tmp_best_transformation_matrix)(0, 3);
		transform_q_t_temp.ty = (tmp_best_transformation_matrix)(1, 3);
		transform_q_t_temp.tz = (tmp_best_transformation_matrix)(2, 3);
		transform_q_t_temp.is_allocate = false;

		high_score tmp_highscore;
		tmp_highscore.qt = transform_q_t_temp;
		tmp_highscore.score = best_score;
		// score_sum += best_score;
		tmp_highest_3score.push_back(tmp_highscore);
	}
	// std::cout << "Totoal score: " << score_sum << std::endl;
	// fuse 
	std::vector<high_score> highest_score;
	for(auto it1 = tmp_highest_3score.begin(); it1 != tmp_highest_3score.end(); it1++){
		if((*it1).score > best_best_score*0.8){
			highest_score.push_back(*it1);
			score_sum += (*it1).score;
		}
	}
	fuse_answer(best_transformation_matrix, highest_score, score_sum);
	// std::cout << best_transformation_matrix << std::endl;
}

void writefile(int pos, std::string filepath, double record){
	int currentpos = 0;
    std::ofstream o2;
    std::string linee;
    std::ifstream fin(filepath, std::ios::in);
    if(fin.good()){
        std::stringstream bufer;
        std::stringstream ss;
        ss << record;
        std::string str = ss.str();
        // std::cout << str << std::endl;
        while (getline(fin, linee)){
            if(currentpos == pos){
                bufer << linee + "," + str << "\n";
            }
            else
                bufer << linee + "\n";
            currentpos++;
        }
        std::ofstream fout(filepath, std::ios::out);
        fout << bufer.rdbuf();
    }
    // else{
    //     std::ofstream o1;
    //     o1.open(filepath2, std::fstream::out);
	// 	o1 << record << std::endl;
	// 	o1.close();
    // }
    if(currentpos < 4 && currentpos == pos){
        std::ofstream o2;
        o2.open(filepath, std::fstream::out | std::fstream::app );
        o2 << record << std::endl;
		o2.close();
    }
}

int main (int argc, char** argv)
{ 
	std::string fnameS = argv[1];
	std::string fnameT = argv[2];
	LeafSize = std::atof(argv[3]);

	pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
	
	if (pcl::io::loadPLYFile<pcl::PointXYZ>(fnameS, *source) == -1)
	{
		PCL_ERROR("Couldn't read file \n");
		return 0;
	}

	if (pcl::io::loadPLYFile<pcl::PointXYZ>(fnameT, *target) == -1)
	{
		PCL_ERROR("Couldn't read file \n");
		return 0;
	}

	std::cout << "Leaf size : " << LeafSize << std::endl;
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_source;
	voxel_grid_source.setLeafSize(LeafSize, LeafSize, LeafSize);
	voxel_grid_source.setInputCloud(source);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
	voxel_grid_source.filter(*cloud_src);
	
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_target;
	voxel_grid_source.setLeafSize(LeafSize, LeafSize, LeafSize);
	voxel_grid_source.setInputCloud(target);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tar(new pcl::PointCloud<pcl::PointXYZ>);
	voxel_grid_source.filter(*cloud_tar);

	Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();
	clock_t a, b;
	a = clock();
	computer_transform_guess(cloud_tar,cloud_src,transformation_matrix);
	b = clock();
	double costTime = double(b-a)/CLOCKS_PER_SEC;

	std::cout << "Transformation: \n" << transformation_matrix << std::endl;
	
	return 0;			
}			