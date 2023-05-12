%% �� ���α׷��� �κ��� �����δ� ����, ���̷δ� ���͸� ���
%% �κ� �հ����� ��� 0~6800 ������ ���� ������.
%% Joint position control�̳� cartesian position control�̳�
%% Joint�� ���� ��ǥ�� �������� �����̳� cartesian�� �����ǥ�� ������
%% �Ѵ� ��ǥ ������ ������ �ٷ� �ǵ��ƿ´ٴ� ������ ����
%% �ڵ忡 ���� ���� �ʴ´ٸ� ������ cartesian velocity/joint velocity�� �Ἥ �����̴� ����� ���� ������.
%% �ݺ��Ǵ� �������� ���� ���ȭ�� ������ ����Ǿ�߰ڴ�.
clc; clear; close;

jc = JacoComm;
connect(jc);
calibrateFingers(jc);

%% Query individual object properties
jc.JointPos
%%
jc.JointVel
%%
jc.JointTorque
%%
jc.JointTemp
%%
jc.FingerPos
%%
jc.FingerVel
%%
jc.FingerTorque
%%
jc.FingerTemp
%%
jc.EndEffectorPose
%%
jc.EndEffectorWrench
%%
jc.ProtectionZone
%%
jc.EndEffectorOffset
%%
jc.DOF
%%
jc.TrajectoryInfo

%% Methods to query joint and finger values all at once
%% �� ���� ���� ���� �հ��� ���� ���� ���� ����
pos = getJointAndFingerPos(jc);
%%
%% �� ���� �ӵ� ���� �հ��� ���� �ӵ� ���� ����
vel = getJointAndFingerVel(jc);
%%
%% �� ���� ��ũ ���� �հ��� ���� ��ũ ���� ����
torque = getJointAndFingerTorque(jc);
temp = getJointAndFingerTemp(jc);

setPositionControlMode(jc);
goToHomePosition(jc);

current_pos=jc.EndEffectorPose;
home_pos=jc.EndEffectorPose;
previous_pos=current_pos;
%home_pos=[0 0 0 0 0 0];
%% Desired_pos�� �۾����� �ٱ��̶�� �ݵ�� ���ܽ��Ѿ� �Ѵ�.
home_pos=jc.EndEffectorPose;
current_pos=jc.EndEffectorPose;
prev_pos=current_pos;

setPositionControlMode(jc);
fCmd = 0*ones(3,1);
sendFingerPositionCommand(jc,fCmd);

desired_pos=[0.7; -0.25; 0.1; home_pos(4); home_pos(5); home_pos(6)];
moveToCP(jc,desired_pos);


pause(1);

setPositionControlMode(jc);
fCmd = 6000*ones(3,1);
sendFingerPositionCommand(jc,fCmd);


desired_pos=[0.7; -0.2; 0.2; home_pos(4); home_pos(5); home_pos(6)];
moveToCP(jc,desired_pos);


Wrist rotation
jntVelCmd = [0;0;0;0;0;0;0.8]; %7DOF
for i=1:260
    sendJointVelocityCommand(jc,jntVelCmd);
end

jntVelCmd = [0;0;0;0;0;0;-0.8]; %7DOF
for i=1:260
    sendJointVelocityCommand(jc,jntVelCmd);
end


desired_pos=[0.7; -0.25; 0.1; home_pos(4); home_pos(5); home_pos(6)];
moveToCP(jc,desired_pos);

pause(1);

setPositionControlMode(jc);
fCmd = 0*ones(3,1);
sendFingerPositionCommand(jc,fCmd);



desired_pos=[0.2; -0.2; 0.4; home_pos(4); home_pos(5); home_pos(6)];
moveToCP(jc,desired_pos);
length=sqrt((desired_pos(1)-current_pos(1)).^2+(desired_pos(2)-current_pos(2)).^2+(desired_pos(3)-current_pos(3)).^2);
CartVel=0.2;
direction=CartVel*[desired_pos(1)-current_pos(1),desired_pos(2)-current_pos(2),desired_pos(3)-current_pos(3)]/length;
time=round(length*200/CartVel);

time=400;

%% i���� �ӵ����� ������ �����ϴ� ���� ����
%% Send cartesian velocity
%% i=200 �� 1�ʿ� �ش�
%% ���� ���� �������� �ڴ� ����. 5ms���� �Է��� ���Ƿ� ���� 0.01�� �Ѵ� ���� �ΰ����� ����
%% ���� ���� ��� �밳 small actuator�� �Ѱ谪�� ������ �־�

error=sqrt((desired_pos(1)-current_pos(1)).^2+(desired_pos(2)-current_pos(2)).^2+(desired_pos(3)-current_pos(3)).^2);

while error>0.01
    temp_pos=jc.EndEffectorPose;
    error=sqrt((desired_pos(1)-temp_pos(1)).^2+(desired_pos(2)-temp_pos(2)).^2+(desired_pos(3)-temp_pos(3)).^2);
    CartVel=0.2;
    direction=CartVel*[desired_pos(1)-temp_pos(1),desired_pos(2)-temp_pos(2),desired_pos(3)-temp_pos(3)]/error;
    CartVelCmd = [direction(1);direction(2);direction(3);0;0;0];
    sendCartesianVelocityCommand(jc,CartVelCmd);
end
for i=1:time
    temp_pos=jc.EndEffectorPose;
    CartVelCmd = [direction(1);direction(2);direction(3);0;0;0];
    sendCartesianVelocityCommand(jc,CartVelCmd);
end

current_pos=jc.EndEffectorPose;

%% ���Ŀ� ���� �������� ������ ���´ٴ��� �ϴ� ������ �����Ͽ��� �Ѵ�.
setPositionControlMode(jc);
fCmd = 0*ones(3,1);
sendFingerPositionCommand(jc,fCmd);

pause(3);

setPositionControlMode(jc);
goToHomePosition(jc);

%% Delete all protection zones
% jc.EraseAllProtectionZones


%% Disconnect from robot and unload library
% goToHomePosition(jc);
% setPositionControlMode(jc);
% disconnect(jc);




%% Online code for KH.Sim
disp('Left');
home_pos=jc.EndEffectorPose;
current_pos=jc.EndEffectorPose;
prev_pos=current_pos;

setPositionControlMode(jc);
fCmd = 0*ones(3,1);
sendFingerPositionCommand(jc,fCmd);
%% Subject home position
desired_pos=[0.45; -0.1; 0.1; home_pos(4); home_pos(5); home_pos(6)];
moveToCP(jc,desired_pos);

setPositionControlMode(jc);
fCmd =0*ones(3,1);
sendFingerPositionCommand(jc,fCmd);

torque_new = getJointAndFingerTorque(jc);
temp_new = getJointAndFingerTemp(jc);
pos_new = getJointAndFingerPos(jc); %���� ���� + �հ��� ����
pause(3);

for k=1:inf
    
    pos_new = getJointAndFingerPos(jc)
    save('trajectory.mat','pos_new')
    pos_new_1 = getJointAndFingerPos(jc)
    save('trajectory.mat','pos_new_1','-append')
    type('trajectory.mat')
end

