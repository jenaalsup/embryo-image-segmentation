// fit ellipsoids to all cells and the entire tissue
void Tissue::compute_ellipsoids()
{
  Point centroid;
  double tissueArea = 0;
  
  for (auto& labeledCell : cells)
  {
    auto& cell = labeledCell.second;
    
    // compute the centroid of the cell area and the area itself
    cell.centroid = Point();
    cell.area = 0;
    for (auto& triangle : cell.triangles)
    {
      Point e1 = points[triangle[1]] - points[triangle[0]];
      Point e2 = points[triangle[2]] - points[triangle[0]];
      Point sum = points[triangle[0]] + points[triangle[1]] + points[triangle[2]];
      double triangleArea = e1.cross(e2).norm() / 2;
      cell.area += triangleArea;
      cell.centroid += sum * (triangleArea/3);
    }
    centroid += cell.centroid;
    tissueArea += cell.area;
    cell.centroid *= 1 / cell.area;
  }
  centroid *= 1 / tissueArea;
  
  Triad I(Point(1,0,0), Point(0,1,0), Point(0,0,1)); // identity matrix
  Triad S(Point(2,1,1), Point(1,2,1), Point(1,1,2));
  Triad tissueInertiaTensor;
  Point principalMoments;
  gte::SymmetricEigensolver3x3<double> eigenSolver;
  for (auto& labeledCell : cells)
  {
    auto& cell = labeledCell.second;
    // compute the intertia tensor of the cell from the inertia tensors of its triangles
    // source: https://ipfs.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Inertia_tensor_of_triangle.html
    Triad cellInertiaTensor;
    for (auto& triangle : cell.triangles)
    {
      Point v0 = points[triangle[0]] - cell.centroid;
      Point v1 = points[triangle[1]] - cell.centroid;
      Point v2 = points[triangle[2]] - cell.centroid;
      Triad V(v0, v1, v2);
      double triangleArea = (v1 - v0).cross(v2 - v0).norm() / 2;
      Triad C = V * S * V.transpose() * (triangleArea / 12);
      
      // inertia tensors about the same point are additive
      cellInertiaTensor += I * C.trace() - C;
    }

    // sum up inertia tensors of all cells using the parallel axis theorem
    Point d = cell.centroid - centroid;
    Triad D(Point(0,d[2],-d[1]), Point(-d[2],0,d[0]), Point(d[1],-d[0],0));
    tissueInertiaTensor += cellInertiaTensor - D * D * cell.area;

    // eigendecomposition (i.e., compute the principal axes and moments of inertia)
    // eigenvalues are sorted in ascending order
    eigenSolver(cellInertiaTensor[0][0], cellInertiaTensor[0][1], cellInertiaTensor[0][2],
                                         cellInertiaTensor[1][1], cellInertiaTensor[1][2],
                                                                  cellInertiaTensor[2][2],
                true, 1,
                principalMoments, cell.ellipse.orientation);
    principalMoments *= 1 / cell.area;
    
    // make sure the third eigenvector always points in positive direction as defined by the cell surface normal,
    // which in turn is assumed to be defined by the orientation of the cell's first triangle
    Triangle& t = cell.triangles.front();
    Point triangleNormal = (points[t[1]] - points[t[0]]).cross(points[t[2]] - points[t[0]]);
    Point cellNormal = cell.ellipse.orientation[2];
    if (triangleNormal.dot(cellNormal) < 0)
    {
      cellNormal *= -1;
      cell.ellipse.orientation[2] = cellNormal;
    }
    
    // compute the semi axes of the mass ellipsoid (multiplied by sqrt(2) such that
    // the mass ellipse of an ellipse is the ellipse itself)
    for (int i = 0; i < 3; ++i)
      cell.ellipse.semiAxes[i] = std::sqrt(2 * (principalMoments[(i+1)%3] + principalMoments[(i+2)%3] - principalMoments[i]));

    // compute the aspect ratio of the cell surface
    cell.ellipse.aspectRatio = cell.ellipse.semiAxes[0] / cell.ellipse.semiAxes[1];
  }
  
  // eigendecomposition of the entire tissue's inertia tensor
  eigenSolver(tissueInertiaTensor[0][0], tissueInertiaTensor[0][1], tissueInertiaTensor[0][2],
                                         tissueInertiaTensor[1][1], tissueInertiaTensor[1][2],
                                                                    tissueInertiaTensor[2][2],
              true, 1,
              principalMoments, orientation);
}
