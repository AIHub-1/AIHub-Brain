function [lf] = ft_compute_leadfield(pos, sens, vol, varargin)

% FT_COMPUTE_LEADFIELD computes a forward solution for a dipole in a a volume
% conductor model. The forward solution is expressed as the leadfield
% matrix (Nchan*3), where each column corresponds with the potential or field
% distributions on all sensors for one of the x,y,z-orientations of the
% dipole.
%
% Use as
%   [lf] = ft_compute_leadfield(pos, sens, vol, ...)
% with input arguments
%   pos    position dipole (1x3 or Nx3)
%   sens   structure with gradiometer or electrode definition
%   vol    structure with volume conductor definition
%
% The vol structure represents a volume conductor model, its contents
% depend on the type of model. The sens structure represents a sensor
% arary, i.e. EEG electrodes or MEG gradiometers.
%
% It is possible to compute a simultaneous forward solution for EEG and MEG
% by specifying sens and grad as two cell-arrays, e.g.
%   sens = {senseeg, sensmeg}
%   vol  = {voleeg,  volmeg}
% This results in the computation of the leadfield of the first element of
% sens and vol, followed by the second, etc. The leadfields of the
% different imaging modalities are concatenated.
%
% Additional input arguments can be specified as key-value pairs, supported
% optional arguments are
%   'reducerank'      = 'no' or number
%   'normalize'       = 'no', 'yes' or 'column'
%   'normalizeparam'  = parameter for depth normalization (default = 0.5)
%   'weight'          = number or 1xN vector, weight for each dipole position (default = 1)
%
% The leadfield weight may be used to specify a (normalized)
% corresponding surface area for each dipole, e.g. when the dipoles
% represent a folded cortical surface with varying triangle size.
%
% Depending on the specific input arguments for the sensor and volume, this
% function will select the appropriate low-level EEG or MEG forward model.
% The leadfield matrix for EEG will have an average reference over all the
% electrodes.
%
% The supported forward solutions for MEG are
%   single sphere (Cuffin and Cohen, 1977)
%   multiple spheres with one sphere per channel (Huang et al, 1999)
%   realistic single shell using superposition of basis functions (Nolte, 2003)
%   leadfield interpolation using a precomputed grid
%   boundary element method (BEM)
%
% The supported forward solutions for EEG are
%   single sphere
%   multiple concentric spheres (up to 4 spheres)
%   leadfield interpolation using a precomputed grid
%   boundary element method (BEM)
%
% See also FT_PREPARE_VOL_SENS, FT_HEADMODEL_ASA, FT_HEADMODEL_BEMCP,
% FT_HEADMODEL_CONCENTRICSPHERES, FT_HEADMODEL_DIPOLI, FT_HEADMODEL_HALFSPACE,
% FT_HEADMODEL_INFINITE, FT_HEADMODEL_LOCALSPHERES, FT_HEADMODEL_OPENMEEG,
% FT_HEADMODEL_SINGLESHELL, FT_HEADMODEL_SINGLESPHERE,
% FT_HEADMODEL_HALFSPACE

% Copyright (C) 2004-2010, Robert Oostenveld
%
% This file is part of FieldTrip, see http://www.ru.nl/neuroimaging/fieldtrip
% for the documentation and details.
%
%    FieldTrip is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    FieldTrip is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with FieldTrip. If not, see <http://www.gnu.org/licenses/>.
%
% $Id: ft_compute_leadfield.m 2775 2011-02-02 22:43:03Z crimic $

persistent warning_issued;

if iscell(sens) && iscell(vol) && numel(sens)==numel(vol)
  % this represents combined EEG and MEG sensors, where each modality has its own volume conduction model
  lf = cell(1,numel(sens));
  for i=1:length(sens)
    lf{i} = ft_compute_leadfield(pos, sens{i}, vol{i}, varargin{:});
  end
  lf = cat(1, lf{:});
  return;
end

if ~isstruct(sens) && size(sens,2)==3
  % definition of electrode positions only, restructure it
  sens = struct('pnt', sens);
end

% determine whether it is EEG or MEG
iseeg = ft_senstype(sens, 'eeg');
ismeg = ft_senstype(sens, 'meg');

% get the optional input arguments
reducerank     = keyval('reducerank', varargin); if isempty(reducerank), reducerank = 'no'; end
normalize      = keyval('normalize' , varargin); if isempty(normalize ), normalize  = 'no'; end
normalizeparam = keyval('normalizeparam', varargin); if isempty(normalizeparam ), normalizeparam = 0.5; end
weight         = keyval('weight', varargin);

% multiple dipoles can be represented either as a 1x(N*3) vector or as a
% as a Nx3 matrix, i.e. [x1 y1 z1 x2 y2 z2] or [x1 y1 z1; x2 y2 z2]
Ndipoles = numel(pos)/3;
if all(size(pos)==[1 3*Ndipoles])
  pos = reshape(pos, 3, Ndipoles)';
end

if isfield(vol, 'unit') && isfield(sens, 'unit') && ~strcmp(vol.unit, sens.unit)
  error('inconsistency in the units of the volume conductor and the sensor array');
end

if ismeg && iseeg
  % this is something that could be implemented relatively easily
  error('simultaneous EEG and MEG not supported');

elseif ~ismeg && ~iseeg
  error('the input does not look like EEG, nor like MEG');

elseif ismeg
  switch ft_voltype(vol)

    case 'singlesphere'
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % MEG single-sphere volume conductor model
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      pnt = sens.pnt; % position of each coil
      ori = sens.ori; % orientation of each coil

      if isfield(vol, 'o')
        % shift dipole and magnetometers to origin of sphere
        pos = pos - repmat(vol.o, Ndipoles, 1);
        pnt = pnt - repmat(vol.o, size(pnt,1), 1);
      end

      if Ndipoles>1
        % loop over multiple dipoles
        lf = zeros(size(pnt,1),3*Ndipoles);
        for i=1:Ndipoles
          lf(:,(3*i-2):(3*i)) = meg_leadfield1(pos(i,:), pnt, ori);
        end
      else
        % only single dipole
        lf = meg_leadfield1(pos, pnt, ori);
      end

      if isfield(sens, 'tra')
        % this appears to be the modern complex gradiometer definition
        % construct the channels from a linear combination of all magnetometers
        lf = sens.tra * lf;
      end

    case 'multisphere'
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % MEG multi-sphere volume conductor model
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      ncoils = length(sens.pnt);

      if size(vol.r, 1)~=ncoils
        error('number of spheres is not equal to the number of coils')
      end

      if size(vol.o, 1)~=ncoils
        error('number of spheres is not equal to the number of coils');
      end

      lf = zeros(ncoils, 3*Ndipoles);
      for chan=1:ncoils
        for dip=1:Ndipoles
          % shift dipole and magnetometer coil to origin of sphere
          dippos = pos(dip,:)       - vol.o(chan,:);
          chnpos = sens.pnt(chan,:) - vol.o(chan,:);
          tmp = meg_leadfield1(dippos, chnpos, sens.ori(chan,:));
          lf(chan,(3*dip-2):(3*dip)) = tmp;
        end
      end

      if isfield(sens, 'tra')
        % this appears to be the modern complex gradiometer definition
        % construct the channels from a linear combination of all magnetometers
        lf = sens.tra * lf;
      end

    case 'neuromag'
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % use external Neuromag toolbox for forward computation
      % this requires that "megmodel" is initialized, which is done in PREPARE_VOL_SENS
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % compute the forward model for all channels
      % tmp1 = ones(1, Ndipoles);
      % tmp2 = 0.01*pos';  %convert to cm
      % lf = megfield([tmp2 tmp2 tmp2],[[1 0 0]'*tmp1 [0 1 0]'*tmp1 [0 0 1]'*tmp1]);
      for dip=1:Ndipoles
        R = 0.01*pos(i,:)'; % convert from cm to m
        Qx = [1 0 0];
        Qy = [0 1 0];
        Qz = [0 0 1];
        lf(:,(3*(dip-1)+1)) = megfield(R, Qx);
        lf(:,(3*(dip-1)+2)) = megfield(R, Qy);
        lf(:,(3*(dip-1)+3)) = megfield(R, Qz);
      end
      % select only those channels from the forward model that are part of the gradiometer definition
      lf = lf(vol.chansel,:);

    case 'nolte'
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % use code from Guido Nolte for the forward computation
      % this requires that "meg_ini" is initialized, which is done in PREPARE_VOL_SENS
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % the dipole position and orientation should be combined in a single matrix
      % furthermore, here I want to compute the leadfield for each of the
      % orthogonzl x/y/z directions
      dippar = zeros(Ndipoles*3, 6);
      for i=1:Ndipoles
        dippar((i-1)*3+1,:) = [pos(i,:) 1 0 0];  % single dipole, x-orientation
        dippar((i-1)*3+2,:) = [pos(i,:) 0 1 0];  % single dipole, y-orientation
        dippar((i-1)*3+3,:) = [pos(i,:) 0 0 1];  % single dipole, z-orientation
      end
      % compute the leadfield for each individual coil
      lf = meg_forward(dippar,vol.forwpar);
      if isfield(sens, 'tra')
        % compute the leadfield for each gradiometer (linear combination of coils)
        lf = sens.tra * lf;
      end

    case 'openmeeg'
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % use code from OpenMEEG
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      ft_hastoolbox('openmeeg', 1);

      dsm = openmeeg_dsm(pos,vol);
      [h2mm,s2mm]= openmeeg_megm(pos,vol,sens);
      
      if isfield(vol,'mat')
        lf = s2mm+h2mm*(vol.mat*dsm);
      else
        error('No system matrix is present, BEM head model not calculated yet')
      end
      if isfield(sens, 'tra')
        % compute the leadfield for each gradiometer (linear combination of coils)
        lf = sens.tra * lf;
      end
      
    case 'infinite'
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % magnetic dipole instead of electric (current) dipole in an infinite vacuum
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if isempty(warning_issued)
        % give the warning only once
        warning('assuming magnetic dipole in an infinite vacuum');
        warning_issued = 1;
      end

      pnt = sens.pnt; % position of each coil
      ori = sens.ori; % orientation of each coil

      if Ndipoles>1
        % loop over multiple dipoles
        lf = zeros(size(pnt,1),3*Ndipoles);
        for i=1:Ndipoles
          lf(:,(3*i-2):(3*i)) = magnetic_dipole(pos(i,:), pnt, ori);
        end
      else
        % only single dipole
        lf = magnetic_dipole(pos, pnt, ori);
      end

      if isfield(sens, 'tra')
        % construct the channels from a linear combination of all magnetometer coils
        lf = sens.tra * lf;
      end

    otherwise
      error('unsupported volume conductor model for MEG');
  end % switch voltype for MEG

elseif iseeg
  switch ft_voltype(vol)

    case 'multisphere'
      % Based on the approximation of the potential due to a single dipole in
      % a multishell sphere by three dipoles in a homogeneous sphere, code
      % contributed by Punita Christopher

      Nelec = size(sens.pnt,1);
      Nspheres = length(vol.r);

      % the center of the spherical volume conduction model does not have
      % to be in the origin, therefore shift the spheres, the electrodes
      % and the dipole
      if isfield(vol, 'o')
        center = vol.o;
      else
        center = [0 0 0];
      end

      % sort the spheres from the smallest to the largest
      % furthermore, the radius should be one (?)
      [radii, indx] = sort(vol.r/max(vol.r));
      sigma = vol.c(indx);
      r   = (sens.pnt-repmat(center, Nelec, 1))./max(vol.r);
      pos = pos./max(vol.r);

      if Ndipoles>1
        % loop over multiple dipoles
        lf = zeros(Nelec,3*Ndipoles);
        for i=1:Ndipoles
          rq = pos(i,:) - center;
          % compute the potential for each dipole ortientation
          % it would be much more efficient to change the punita function
          q1 = [1 0 0]; lf(:,(3*i-2)) = multisphere(Nspheres, radii, sigma, r, rq, q1);
          q1 = [0 1 0]; lf(:,(3*i-1)) = multisphere(Nspheres, radii, sigma, r, rq, q1);
          q1 = [0 0 1]; lf(:,(3*i  )) = multisphere(Nspheres, radii, sigma, r, rq, q1);
        end
      else
        % only single dipole
        lf = zeros(Nelec,3);
        rq = pos - center;
        % compute the potential for each dipole ortientation
        % it would be much more efficient to change the punita function
        q1 = [1 0 0] ; lf(:,1) = multisphere(Nspheres, radii, sigma, r, rq, q1);
        q1 = [0 1 0] ; lf(:,2) = multisphere(Nspheres, radii, sigma, r, rq, q1);
        q1 = [0 0 1] ; lf(:,3) = multisphere(Nspheres, radii, sigma, r, rq, q1);
      end

    case {'singlesphere', 'concentric'}
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % EEG spherical volume conductor model
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      % FIXME, this is not consistent between spherical and BEM
      % sort the spheres from the smallest to the largest
      [vol.r, indx] = sort(vol.r);
      vol.c = vol.c(indx);

      Nspheres = length(vol.c);
      if length(vol.r)~=Nspheres
        error('the number of spheres in the volume conductor model is ambiguous');
      end

      if isfield(vol, 'o')
        % shift the origin of the spheres, electrodes and dipole
        sens.pnt = sens.pnt - repmat(vol.o, size(sens.pnt,1), 1);
        pos = pos - repmat(vol.o, Ndipoles, 1);
      end

      switch Nspheres
        case 1
          funnam = 'eeg_leadfield1';
        case 2
          vol.r = [vol.r(1) vol.r(2) vol.r(2) vol.r(2)];
          vol.c = [vol.c(1) vol.c(2) vol.c(2) vol.c(2)];
          funnam = 'eeg_leadfield4';
        case 3
          vol.r = [vol.r(1) vol.r(2) vol.r(3) vol.r(3)];
          vol.c = [vol.c(1) vol.c(2) vol.c(3) vol.c(3)];
          funnam = 'eeg_leadfield4';
        case 4
          vol.r = [vol.r(1) vol.r(2) vol.r(3) vol.r(4)];
          vol.c = [vol.c(1) vol.c(2) vol.c(3) vol.c(4)];         
          funnam = 'eeg_leadfield4';
        otherwise
          error('more than 4 concentric spheres are not supported')
      end
      
      lf = zeros(size(sens.pnt,1),3*Ndipoles);
      for i=1:Ndipoles
        lf(:,(3*i-2):(3*i)) = feval(funnam,pos(i,:), sens.pnt, vol);
      end

    case {'bem', 'dipoli', 'asa', 'avo', 'bemcp'}
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % EEG boundary element method volume conductor model
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      lf = eeg_leadfieldb(pos, sens.pnt, vol);
    
    case 'openmeeg'
      if ft_hastoolbox('openmeeg', 1);
        dsm = openmeeg_dsm(pos,vol);
        if isfield(vol,'mat')
          lf = vol.mat*dsm;
        else
          error('No system matrix is present, BEM head model not calculated yet')
        end
      else
        error('Openmeeg toolbox not installed')
      end
  
      case 'metufem'
        p3 = zeros(Ndipoles * 3, 6);
        for i = 1:Ndipoles
          p3((3*i - 2) : (3 * i), 1:3) = [pos(i,:); pos(i,:); pos(i,:)];
          p3((3*i - 2) : (3 * i), 4:6) = [1 0 0; 0 1 0; 0 0 1];
        end
        lf = metufem('pot', p3');

      case 'metubem'
        session = vol.session;
        p3 = zeros(Ndipoles * 3, 6);
        for i = 1:Ndipoles
          p3((3*i - 2) : (3 * i), 1:3) = [pos(i,:); pos(i,:); pos(i,:)];
          p3((3*i - 2) : (3 * i), 4:6) = [1 0 0; 0 1 0; 0 0 1];
        end
        [lf, session] = bem_solve_lfm_eeg(session, p3);

    case 'infinite'
      % the conductivity of the medium is not known
      if isempty(warning_issued)
        % give the warning only once
        warning('assuming electric dipole in an infinite medium with unit conductivity');
        warning_issued = 1;
      end
      lf = inf_medium_leadfield(pos, sens.pnt, 1);
  
    case 'halfspace'
      lf = eeg_halfspace_medium_leadfield(pos, sens.pnt, vol);

    otherwise
      error('unsupported volume conductor model for EEG');
  end % switch voltype for EEG

  % compute average reference for EEG leadfield
  avg = mean(lf, 1);
  lf  = lf - repmat(avg, size(lf,1), 1);
  % apply the correct montage to the leadfield
  if isfield(sens,'tra')
    lf = sens.tra*lf;
  end
  
end % iseeg or ismeg

% optionally apply leadfield rank reduction
if ~strcmp(reducerank, 'no') && reducerank<size(lf,2) && ~strcmp(ft_voltype(vol),'openmeeg')
  % decompose the leadfield
  [u, s, v] = svd(lf);
  r = diag(s);
  s(:) = 0;
  for j=1:reducerank
    s(j,j) = r(j);
  end
  % recompose the leadfield with reduced rank
  lf = u * s * v';
end

% optionally apply leadfield normaliziation
switch normalize
case 'yes'
  if normalizeparam==0.5
    % normalize the leadfield by the Frobenius norm of the matrix
    % this is the same as below in case normalizeparam is 0.5
    nrm = norm(lf, 'fro');
  else
    % normalize the leadfield by sum of squares of the elements of the leadfield matrix to the power "normalizeparam"
    % this is the same as the Frobenius norm if normalizeparam is 0.5
    nrm = sum(lf(:).^2)^normalizeparam;
  end
  if nrm>0
    lf = lf ./ nrm;
  end
case 'column'
  % normalize each column of the leadfield by its norm
  for j=1:size(lf,2)
    nrm = sum(lf(:,j).^2)^normalizeparam;
    lf(:,j) = lf(:,j)./nrm;
  end
end

% optionally apply a weight to the leadfield for each dipole location
if ~isempty(weight)
  for i=1:Ndipoles
    lf(:,3*(i-1)+1) = lf(:,3*(i-1)+1) * weight(i); % the leadfield for the x-direction
    lf(:,3*(i-1)+2) = lf(:,3*(i-2)+1) * weight(i); % the leadfield for the y-direction
    lf(:,3*(i-1)+3) = lf(:,3*(i-3)+1) * weight(i); % the leadfield for the z-direction
  end
end
  