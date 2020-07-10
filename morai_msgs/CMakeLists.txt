cmake_minimum_required(VERSION 2.8.3)
project(morai_msgs)


find_package(catkin REQUIRED COMPONENTS
  std_msgs
  message_generation
)

 add_message_files(
   DIRECTORY msg
   FILES
   CtrlCmd.msg
   EgoVehiclePosition.msg
   EgoVehicleStatus.msg
   EgoVehicleVelocity.msg
   FactoryAddress.msg
   GPSMessage.msg
   GuideData.msg
   ObjectInfo.msg
   ScenarioLoad.msg
   SVehicleStatus.msg
   TrafficLight.msg
   VehicleCommand.msg
   VelPlot.msg
   ERP42Info.msg
 )

 generate_messages(
   DEPENDENCIES
   std_msgs
 )

catkin_package(
  CATKIN_DEPENDS 
std_msgs 
message_runtime
)

