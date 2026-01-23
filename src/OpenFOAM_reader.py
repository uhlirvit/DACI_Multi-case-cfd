"""
OpenFOAM Data Reader with Profile Averaging
Extracts clean vertical profiles from OpenFOAM data
"""

import numpy as np
from pathlib import Path

class OpenFOAMReader:
    """Read OpenFOAM case data"""

    def __init__(self, case_path):
        """
        Initialize reader

        Args:
            case_path: Path to OpenFOAM case directory
        """
        self.case_path = Path(case_path)
        self.mesh = None
        self.cell_centers = None

    def read_cell_centers(self, time='latestTime'):
        """
        Read cell center coordinates

        Args:
            time: Time directory (or 'latestTime' for most recent)

        Returns:
            cell_centers: numpy array [n_cells, 3] with (x, y, z) coordinates
        """
        # Find time directory
        if time == 'latestTime':
            time_dirs = [d for d in self.case_path.iterdir()
                        if d.is_dir() and d.name.replace('.', '').replace('-', '').isdigit()]
            time_dirs.sort(key=lambda x: float(x.name))
            time_dir = time_dirs[-1]
        else:
            time_dir = self.case_path / str(time)

        # Read cell centers file
        cc_file = time_dir / 'C'

        if not cc_file.exists():
            print("\n" + "="*70)
            print("ERROR: Cell centers file not found!")
            print("="*70)
            print(f"Looking for: {cc_file}")
            print("\nPlease run in your OpenFOAM case directory:")
            print(f"  postProcess -func writeCellCentres -time {time_dir.name}")
            print("\nOR for latest time:")
            print("  postProcess -func writeCellCentres -latestTime")
            print("="*70)
            return None

        print(f"Reading cell centers from: {cc_file}")
        cell_centers = self._read_vector_field(cc_file)

        self.cell_centers = cell_centers
        return cell_centers

    def read_vector_field(self, field_name, time='latestTime'):
        """
        Read a vector field (like U)

        Args:
            field_name: Name of field ('U', 'p', etc.)
            time: Time directory

        Returns:
            field_data: numpy array [n_cells, 3] for vector or [n_cells] for scalar
        """
        # Find time directory
        if time == 'latestTime':
            time_dirs = [d for d in self.case_path.iterdir()
                        if d.is_dir() and d.name.replace('.', '').replace('-', '').isdigit()]
            time_dirs.sort(key=lambda x: float(x.name))
            time_dir = time_dirs[-1]
        else:
            time_dir = self.case_path / str(time)

        field_file = time_dir / field_name

        if not field_file.exists():
            raise FileNotFoundError(f"Field file not found: {field_file}")

        print(f"Reading field from: {field_file}")

        # Determine if vector or scalar
        with open(field_file, 'r') as f:
            content = f.read()
            if 'volVectorField' in content:
                return self._read_vector_field(field_file)
            elif 'volScalarField' in content:
                return self._read_scalar_field(field_file)

    def _read_vector_field(self, file_path):
        """Read vector field from OpenFOAM file"""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Find start of internalField data
        start_idx = None
        n_cells = None

        for i, line in enumerate(lines):
            if 'internalField' in line and 'nonuniform' in line:
                if 'List<vector>' in line or 'List<scalar>' in line:
                    if i + 1 < len(lines):
                        try:
                            n_cells = int(lines[i + 1].strip())
                            start_idx = i + 3
                        except ValueError:
                            parts = line.split()
                            for part in parts:
                                try:
                                    n_cells = int(part)
                                    break
                                except ValueError:
                                    continue
                            start_idx = i + 2
                break

        if start_idx is None or n_cells is None:
            raise ValueError("Could not parse OpenFOAM field file")

        print(f"Reading {n_cells:,} cells...")

        # Read vector data
        vectors = []
        for i in range(start_idx, start_idx + n_cells):
            if i >= len(lines):
                break

            line = lines[i].strip()
            if line.startswith('(') and line.endswith(')'):
                line = line[1:-1]
                components = line.split()
                if len(components) == 3:
                    vector = [float(c) for c in components]
                    vectors.append(vector)

        return np.array(vectors)

    def _read_scalar_field(self, file_path):
        """Read scalar field from OpenFOAM file"""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Find start of internalField data
        start_idx = None
        n_cells = None

        for i, line in enumerate(lines):
            if 'internalField' in line and 'nonuniform' in line:
                if i + 1 < len(lines):
                    try:
                        n_cells = int(lines[i + 1].strip())
                        start_idx = i + 3
                    except ValueError:
                        parts = line.split()
                        for part in parts:
                            try:
                                n_cells = int(part)
                                break
                            except ValueError:
                                continue
                        start_idx = i + 2
                break

        if start_idx is None or n_cells is None:
            raise ValueError("Could not parse OpenFOAM field file")

        # Read scalar data
        scalars = []
        for i in range(start_idx, start_idx + n_cells):
            if i >= len(lines):
                break

            line = lines[i].strip()
            try:
                value = float(line)
                scalars.append(value)
            except ValueError:
                continue

        return np.array(scalars)

    def extract_vertical_profile(self, field_data, x_target, y_target=None,
                                 tolerance=0.01, method='average', n_bins=100):
        """
        Extract vertical profile at specified x (and optionally y) location

        Args:
            field_data: Field data array
            x_target: Target x coordinate
            y_target: Target y coordinate (None for 2D/any y)
            tolerance: Distance tolerance for point selection
            method: 'average' (average at each height),
                   'closest' (take closest point at each height),
                   'all' (return all points - raw cloud)
            n_bins: Number of height bins for averaging (if method='average')

        Returns:
            z_coords: z coordinates of profile
            field_values: field values at those z coordinates
        """
        if self.cell_centers is None:
            raise ValueError("Must read cell centers first!")

        # Extract points near target x (and y if specified)
        x = self.cell_centers[:, 0]
        y = self.cell_centers[:, 1]
        z = self.cell_centers[:, 2]

        # Find cells near target location
        if y_target is not None:
            mask = (np.abs(x - x_target) < tolerance) & \
                   (np.abs(y - y_target) < tolerance)
        else:
            mask = np.abs(x - x_target) < tolerance

        # Extract profile data
        x_selected = x[mask]
        z_selected = z[mask]

        if len(field_data.shape) == 2:
            field_selected = field_data[mask, :]
        else:
            field_selected = field_data[mask]

        print(f"\nExtracted {len(z_selected)} points near x={x_target}")
        if y_target is not None:
            print(f"  and y={y_target}")

        # Process based on method
        if method == 'all':
            # Return all points (original behavior - point cloud)
            sort_idx = np.argsort(z_selected)
            return z_selected[sort_idx], field_selected[sort_idx]

        elif method == 'closest':
            # At each z-level, keep only closest point to x_target
            return self._extract_closest_points(
                x_selected, z_selected, field_selected, x_target
            )

        elif method == 'average':
            # Average all points at each height bin
            return self._extract_averaged_profile(
                z_selected, field_selected, n_bins
            )

        else:
            raise ValueError(f"Unknown method: {method}. Use 'average', 'closest', or 'all'")

    def _extract_closest_points(self, x_selected, z_selected, field_selected, x_target):
        """
        For each z-level, keep only the point closest to x_target
        """
        # Create height bins to group points
        z_min, z_max = z_selected.min(), z_selected.max()
        n_bins = min(200, len(z_selected) // 2)  # Adaptive binning

        if n_bins < 10:
            n_bins = len(np.unique(z_selected))

        z_bins = np.linspace(z_min, z_max, n_bins + 1)

        z_profile = []
        field_profile = []

        for i in range(n_bins):
            # Find points in this height bin
            mask_bin = (z_selected >= z_bins[i]) & (z_selected < z_bins[i + 1])

            if mask_bin.sum() == 0:
                continue

            # Find closest point to x_target in this bin
            x_in_bin = x_selected[mask_bin]
            distances = np.abs(x_in_bin - x_target)
            closest_idx = np.argmin(distances)

            # Get data for closest point
            z_in_bin = z_selected[mask_bin]
            field_in_bin = field_selected[mask_bin]

            z_profile.append(z_in_bin[closest_idx])

            if len(field_in_bin.shape) == 2:
                field_profile.append(field_in_bin[closest_idx, :])
            else:
                field_profile.append(field_in_bin[closest_idx])

        z_profile = np.array(z_profile)
        field_profile = np.array(field_profile)

        # Sort by height
        sort_idx = np.argsort(z_profile)

        print(f"  Reduced to {len(z_profile)} points (closest at each height)")

        return z_profile[sort_idx], field_profile[sort_idx]

    def _extract_averaged_profile(self, z_selected, field_selected, n_bins):
        """
        Average field values at each height bin
        """
        z_min, z_max = z_selected.min(), z_selected.max()
        z_bins = np.linspace(z_min, z_max, n_bins + 1)

        z_profile = []
        field_profile = []

        for i in range(n_bins):
            # Find points in this height bin
            mask_bin = (z_selected >= z_bins[i]) & (z_selected < z_bins[i + 1])

            if mask_bin.sum() == 0:
                continue

            # Average z and field values in this bin
            z_profile.append(z_selected[mask_bin].mean())

            if len(field_selected.shape) == 2:
                # Vector field - average each component
                field_profile.append(field_selected[mask_bin, :].mean(axis=0))
            else:
                # Scalar field
                field_profile.append(field_selected[mask_bin].mean())

        z_profile = np.array(z_profile)
        field_profile = np.array(field_profile)

        print(f"  Averaged into {len(z_profile)} height bins")
        print(f"  z range: [{z_profile.min():.6f}, {z_profile.max():.6f}] m")

        return z_profile, field_profile


def extract_profile_from_openfoam(case_path, x_location, time='latestTime',
                                  y_location=None, tolerance=0.01,
                                  method='average', n_bins=100):
    """
    Convenience function to extract vertical profile

    Args:
        case_path: Path to OpenFOAM case
        x_location: x coordinate for profile
        time: Time directory or 'latestTime'
        y_location: y coordinate (None for 2D/any y)
        tolerance: Search tolerance
        method: 'average' (recommended), 'closest', or 'all'
        n_bins: Number of height bins for averaging

    Returns:
        z, U_x, U_y, U_z: Profile data (clean arrays)
    """
    reader = OpenFOAMReader(case_path)

    print("="*70)
    print("READING OPENFOAM DATA")
    print("="*70)

    # Read cell centers
    cell_centers = reader.read_cell_centers(time)

    if cell_centers is None:
        return None, None, None, None

    # Read velocity field
    U_field = reader.read_vector_field('U', time)

    # Extract profile
    z, U_profile = reader.extract_vertical_profile(
        U_field, x_location, y_location, tolerance, method, n_bins
    )

    # Separate velocity components
    U_x = U_profile[:, 0]
    U_y = U_profile[:, 1]
    U_z = U_profile[:, 2]

    print("\n" + "="*70)
    print("PROFILE EXTRACTED")
    print("="*70)
    print(f"Method: {method}")
    print(f"Points in profile: {len(z)}")
    print(f"z range: [{z.min():.6f}, {z.max():.6f}] m")
    print(f"U_x range: [{U_x.min():.6f}, {U_x.max():.6f}] m/s")
    print("="*70 + "\n")

    return z, U_x, U_y, U_z


# Convenience functions for specific methods

def extract_averaged_profile(case_path, x_location, time='latestTime',
                             tolerance=0.01, n_bins=100):
    """
    Extract clean averaged vertical profile (RECOMMENDED)
    """
    return extract_profile_from_openfoam(
        case_path, x_location, time,
        y_location=None, tolerance=tolerance,
        method='average', n_bins=n_bins
    )


def extract_closest_profile(case_path, x_location, time='latestTime',
                           tolerance=0.01):
    """
    Extract profile using closest points at each height
    """
    return extract_profile_from_openfoam(
        case_path, x_location, time,
        y_location=None, tolerance=tolerance,
        method='closest'
    )


def extract_raw_profile(case_path, x_location, time='latestTime',
                       tolerance=0.01):
    """
    Extract raw point cloud (original behavior)
    """
    return extract_profile_from_openfoam(
        case_path, x_location, time,
        y_location=None, tolerance=tolerance,
        method='all'
    )
