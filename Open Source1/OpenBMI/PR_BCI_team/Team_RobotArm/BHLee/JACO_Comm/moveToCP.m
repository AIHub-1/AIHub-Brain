%% ������ǥ�� ����� ������ǥ�� �󿡼��� point by point move �Լ�
%% ����� ��ǥ pose�� ��ġ ���ް� �ڼ� ��� ���ÿ� ����

function [stat] = moveToCP(obj,desired_pos)

% Input size üũ
if (size(desired_pos)~=6)
    disp('Check the size of input');
    stat=-1;
    return;
end    
    
% Cartesian ������ �ִ� �۾������� �Ѿ���� üũ
limit_x=0.9;
limit_y=0.9;
limit_z=1.1;
current_pos=obj.EndEffectorPose;

if abs(desired_pos(1))>limit_x
    disp('Unable to move');
    stat=-1;
    return;
end    

if abs(desired_pos(2))>limit_y
    disp('Unable to move');
     stat=-1;
end

if (desired_pos(3)>limit_z || desired_pos(3)<0) 
    stat=-1;
    disp('Unable to move');
end


%���� ������ �ʰ����� �ʵ��� ����
%������ ��� -pi���� pi�� ������ ������ ��

if abs(desired_pos(4))>pi
   desired_pos(4)=mod(desired_pos(4)+pi,2*pi)-pi;
end    

if abs(desired_pos(5))>limit_y
   desired_pos(5)=mod(desired_pos(5)+pi,2*pi)-pi;
end

if abs(desired_pos(6))>limit_y
   desired_pos(6)=mod(desired_pos(6)+pi,2*pi)-pi;
end
% i���� �ӵ����� ������ �����ϴ� ���� ���� 
% Send cartesian velocity
% i=200 �� 1�ʿ� �ش�
% ���� ���� �������� �ڴ� ����. 5ms���� �Է��� ���Ƿ� ���� 0.01�� �Ѵ� ���� �ΰ����� ����
% ���� ���� ��� �밳 small actuator�� �Ѱ谪�� ������ �־� 
% �ڿ� xyz�� ���� ���� ���� ������ 
% error�� 1cm �̸����� ���鵵�� ��


tollerance_xyz=0.01;
%tollerance_angle=0.3;

error_xyz=sqrt((desired_pos(1)-current_pos(1)).^2+(desired_pos(2)-current_pos(2)).^2+(desired_pos(3)-current_pos(3)).^2);
%error_angle=sqrt((desired_pos(4)-current_pos(4)).^2+(desired_pos(5)-current_pos(5)).^2+(desired_pos(6)-current_pos(6)).^2);
timestep=0;
while error_xyz>tollerance_xyz 
    temp_pos=obj.EndEffectorPose;
    
    %% xyz���� ���� �Ÿ� ��
    error_xyz=sqrt((desired_pos(1)-temp_pos(1)).^2+(desired_pos(2)-temp_pos(2)).^2+(desired_pos(3)-temp_pos(3)).^2);
    CartVel=0.2;
    direction=CartVel*[desired_pos(1)-temp_pos(1),desired_pos(2)-temp_pos(2),desired_pos(3)-temp_pos(3)]/error_xyz;
    
%     if(error_xyz<0.1)
%          CartVel=0.9*error_xyz;
%     end
    
%     error_pose=sqrt(abs(mod(desired_pos(4)-temp_pos(4),2*pi)-2*pi).^2+abs(mod(desired_pos(5)-temp_pos(5),2*pi)-2*pi).^2+abs(mod(desired_pos(6)-temp_pos(6),2*pi)-2*pi).^2);
%     % xyz���� ���� ���� ��.
%     AngleVel=0.1;
%     pose=AngleVel*[desired_pos(4)-temp_pos(4),desired_pos(5)-temp_pos(5),desired_pos(6)-temp_pos(6)]/error_pose;
%     
%     if(error_pose<1)
%          AngleVel=0.1*error_pose;
%     end
%         
%    CartVelCmd = [direction(1);direction(2);direction(3);pose(1);pose(2);pose(3)];
    CartVelCmd = [direction(1);direction(2);direction(3);0;0;0];
    sendCartesianVelocityCommand(obj,CartVelCmd);
    
    timestep=timestep+1;
    if(timestep>2000)
        break;
    end    
end



stat=0;
end

