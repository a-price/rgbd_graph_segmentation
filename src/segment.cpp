/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#include <cstdio>
#include <cstdlib>
#include "misc.h"
#include "pnmfile.h"
#include "filter.h"
#include "segment-graph.h"
#include "felz.h"


namespace felzenszwalb
{

// dissimilarity measure between pixels
static inline float diff(image<float> *r, image<float> *g, image<float> *b,
			 int x1, int y1, int x2, int y2) {
  return sqrt(square(imRef(r, x1, y1)-imRef(r, x2, y2)) +
	      square(imRef(g, x1, y1)-imRef(g, x2, y2)) +
	      square(imRef(b, x1, y1)-imRef(b, x2, y2)));
}


struct DepthEdgeFinder
{
  DepthEdgeFinder (image<rgbd>* im, float depth_threshold) :
    im(im), t(depth_threshold)
  {}

  bool operator() (int x, int y, int x2, int y2) const
  {
    const float d = imRef(im, x, y).d;
    const float d2 = imRef(im, x2, y2).d;
    if (!std::isfinite(d))
      return std::isfinite(d2);
    else if (!std::isfinite(d2))
      return true;
    else return fabs(d-d2)>t;
  }

  image<rgbd>* im;
  float t;
};


/*
 * Segment an rgbd image
 *
 * Returns a color image representing the segmentation.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for thkreshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 * depth_threshold: we won't consider a pair of pixels if the depth difference
 * exceeds this
 */
image<int>* segment_image(image<rgbd> *im, float sigma, float c, int min_size,
			  float depth_threshold, int *num_ccs) {
  int width = im->width();
  int height = im->height();

  image<float>* r = new image<float>(width, height);
  image<float>* g = new image<float>(width, height);
  image<float>* b = new image<float>(width, height);

  // smooth each color channel, extract depths
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      imRef(r, x, y) = imRef(im, x, y).r;
      imRef(g, x, y) = imRef(im, x, y).g;
      imRef(b, x, y) = imRef(im, x, y).b;
    }
  }
  image<float> *smooth_r = smooth(r, sigma);
  image<float> *smooth_g = smooth(g, sigma);
  image<float> *smooth_b = smooth(b, sigma);
  delete r;
  delete g;
  delete b;

  DepthEdgeFinder depth_edge(im, depth_threshold);
 
  // build graph
  edge *edges = new edge[width*height*4];
  int num = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x < width-1 && !depth_edge(x, y, x+1, y)) {
          edges[num].a = y * width + x;
          edges[num].b = y * width + (x+1);
          edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y);
          num++;
      }

      if (y < height-1 && !depth_edge(x, y, x, y+1)) {
	edges[num].a = y * width + x;
	edges[num].b = (y+1) * width + x;
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y+1);
	num++;
      }

      if ((x < width-1) && (y<height-1) && !depth_edge(x, y, x+1, y+1)) {
	edges[num].a = y * width + x;
	edges[num].b = (y+1) * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1);
	num++;
      }

      if ((x < width-1) && (y>0) && !depth_edge(x, y, x+1, y-1)) {
	edges[num].a = y * width + x;
	edges[num].b = (y-1) * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1);
	num++;
      }
    }
  }
  delete smooth_r;
  delete smooth_g;
  delete smooth_b;

  // segment
  universe *u = segment_graph(width*height, num, edges, c);
  
  // post process small components
  for (int i = 0; i < num; i++) {
    int a = u->find(edges[i].a);
    int b = u->find(edges[i].b);
    if ((a != b) &&
        ((u->size(a) < min_size) || (u->size(b) < min_size)))
      u->join(a, b);
  }
  delete [] edges;
  *num_ccs = u->num_sets();

  image<int> *output = new image<int>(width, height);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int comp = u->find(y * width + x);
      imRef(output, x, y) = comp;
    }
  }  

  delete u;

  return output;
}

} // namespace


#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace felzenszwalb
{

static inline Eigen::Matrix<float, 9, 1> getParamVector(pcl::PointXYZRGBNormal p)
{
	Eigen::Matrix<float, 9, 1> p_vec;
	p_vec << p.getVector3fMap()/10 , p.getRGBVector3i().cast<float>()/255.0 /3.0 , p.getNormalVector3fMap().normalized() * 10.0;
	//p_vec << p.getVector3fMap() , p.getRGBVector3i().cast<float>()/255.0 , p.getNormalVector3fMap().normalized() * 10.0;
	return p_vec;
}

static float diff(pcl::PointXYZRGBNormal& a, pcl::PointXYZRGBNormal& b)
{
	Eigen::VectorXf a_vec = getParamVector(a);
	Eigen::VectorXf b_vec = getParamVector(b);
	return (a_vec - b_vec).norm();
}

struct pclDepthEdgeFinder
{
	pclDepthEdgeFinder (image<pcl::PointXYZRGBNormal>* im, float depth_threshold) :
		im(im), t(depth_threshold)
	{}

	bool operator() (int x, int y, int x2, int y2) const
	{
		Eigen::Vector3f p1 = imRef(im, x, y).getVector3fMap();
		Eigen::Vector3f p2 = imRef(im, x2, y2).getVector3fMap();
		if (!std::isfinite(p1.x()))
			return std::isfinite(p2.x());
		else if (!std::isfinite(p2.x()))
			return true;
		else return (p1-p2).norm()>t;
	}

	image<pcl::PointXYZRGBNormal>* im;
	float t;
};

/*
 * Segment an rgbd image
 *
 * Returns a color image representing the segmentation.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for threshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 * depth_threshold: we won't consider a pair of pixels if the depth difference
 * exceeds this
 */
image<int>* segment_image(image<pcl::PointXYZRGBNormal>* im, float sigma, float c, int min_size,
						  float depth_threshold, int *num_ccs) {
	int width = im->width();
	int height = im->height();

	pclDepthEdgeFinder depth_edge(im, depth_threshold);

	// build graph
	edge *edges = new edge[width*height*4];
	int num = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int a = y * width + x;
			// Check to the right
			if (x < width-1 && !depth_edge(x, y, x+1, y))
			{
				int b = y * width + (x+1);
				edges[num].a = a;
				edges[num].b = b;
				edges[num].w = diff(im->data[a], im->data[b]);
				num++;
			}

			// Check down
			if (y < height-1 && !depth_edge(x, y, x, y+1))
			{
				int b = (y+1) * width + x;
				edges[num].a = a;
				edges[num].b = b;
				edges[num].w = diff(im->data[a], im->data[b]);
				num++;
			}

			// Check down+right
			if ((x < width-1) && (y<height-1) && !depth_edge(x, y, x+1, y+1))
			{
				int b = (y+1) * width + (x+1);
				edges[num].a = a;
				edges[num].b = b;
				edges[num].w = diff(im->data[a], im->data[b]);
				num++;
			}

			// Check down+left
			if ((x < width-1) && (y>0) && !depth_edge(x, y, x+1, y-1))
			{
				int b = (y-1) * width + (x+1);
				edges[num].a = a;
				edges[num].b = b;
				edges[num].w = diff(im->data[a], im->data[b]);
				num++;
			}
		}
	}

	// segment
	universe *u = segment_graph(width*height, num, edges, c);

	// post process small components
	for (int i = 0; i < num; i++)
	{
		int a = u->find(edges[i].a);
		int b = u->find(edges[i].b);
		if ((a != b) &&
				((u->size(a) < min_size) || (u->size(b) < min_size)))
			u->join(a, b);
	}
	delete [] edges;
	*num_ccs = u->num_sets();

	image<int> *output = new image<int>(width, height);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int comp = u->find(y * width + x);
			imRef(output, x, y) = comp;
		}
	}

	delete u;

	return output;
}

}
