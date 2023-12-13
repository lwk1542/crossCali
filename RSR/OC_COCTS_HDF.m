% OC_COCTS_HDF


clear;
path='/Volumes/Song_HY1B/HY1B/L1b';
path='';
% HFil='H1BCLD090226034809803.L1B.HDF';
HFil='H1BCLR090529031011122.L1B.HDF';


PathFil=fullfile(path,HFil);
ProducLev=PathFil(end-6:end-4);%自动截取数据产品级别  'L1B','L2A'等
% PathFil='/Volumes/Seagate Backup Plus Drive/123/L1b/H1BCBD090224012199999.L1B.HDF'
                
% L_490 = hdfread('/Volumes/Seagate Backup Plus Drive/123/L1b/H1BCLD090207135809541.L1B.HDF', '/Geophysical Data/L_490', 'Index', {[1  1],[1  1],[6304  1664]});




Finfo=hdfinfo(PathFil);
% X=Finfo.Vgroup(5).SDS(:).Name;
pin=[];

for k=1:length(Finfo.Attributes)
  nm=Finfo.Attributes(k).Name;nm(nm==' ')='_';
  if isstr(Finfo.Attributes(k).Value),
    pin=setfield(pin,nm,Finfo.Attributes(k).Value);
  else
    pin=setfield(pin,nm,double(Finfo.Attributes(k).Value));
  end

end% lon/lat of grid corners

%{'Upper Left Latitude';
% 'Upper Left Longitude';
% 'Upper Right Latitude';
% 'Upper Right Longitude';
% 'Lower Left Latitude';
% 'Lower Left Longitude';
% 'Lower Right Latitude';
% 'Lower Right Longitude';
% 'Northernmost Latitude';
% 'Southernmost Latitude';
% 'Westernmost Longitude';
% 'Easternmost Longitude'}

pin.Upper_Left_Longitude;
pin.Upper_Right_Longitude;
pin.Lower_Left_Longitude;
pin.Lower_Right_Longitude;
Vertex_Lon=[pin.Upper_Left_Longitude,...
            pin.Upper_Right_Longitude,...
            pin.Lower_Left_Longitude,...
            pin.Lower_Right_Longitude];
%角点经度差大于180度，图像过180度

L865=double(hdfread(PathFil,'L_865'));
L412=double(hdfread(PathFil,'L_412'));
L520=double(hdfread(PathFil,'L_520'));

lat=hdfread(PathFil,'Latitude');
lon=hdfread(PathFil,'Longitude');

switch ProducLev
    case 'L1B'
        %经纬度插值
        xi=7:10:1657;%7到1657个点，每10个点一个控制点
        
%         [m n]=size(lat);
%         XXXi=repmat(xi,m,1);
%         xq=repmat(7:1657,m,1);
        lat2=[];
        lon2=[];
        if max(Vertex_Lon)-min(Vertex_Lon)>180%图像跨180度经线
            inxlon=(lon<0 & lon>=-180);
            lon(inxlon)=360+lon(inxlon);
         end
        lat2=interp1(xi,lat',7:1657);%内插
        lon2=interp1(xi,lon',7:1657);
        lat_head=interp1(xi,lat',1:6,'linear','extrap');%外插
        lat_end=interp1(xi,lat',1658:1664,'linear','extrap');%外插
        lon_head=interp1(xi,lon',1:6,'linear','extrap');
        lon_end=interp1(xi,lon',1658:1664,'linear','extrap');
        LON=cat(1,lon_head,lon2,lon_end);%拼接成1664个点阵
        LAT=cat(1,lat_head,lat2,lat_end);
        
       LON=LON';%转置
       LAT=LAT';
    case {'L2A','L2B'}
        LON=lon;%不用插值，直接传递过来
        LAT=lat;
end


%画图 投影
LATLIMS=double([min(LAT(:)) max(LAT(:))]);
LONLIMS=double([min(LON(:)) max(LON(:))]);

g(:,:,1)=L865/max(L865(:));
g(:,:,2)=L520/max(L520(:));
g(:,:,3)=L412/max(L412(:));
% g=uint8(g);

figure;imagesc(g);

% [IND,map] = rgb2ind(g,256,'nodither');
% imagesc(g)
% M=rgb2hsv(g);
% figure;
% pcolor(LON,LAT,M(:,:,1));shading interp;colormap(map);
% C=g(:,:,1).^2+g(:,:,2).^2+g(:,:,3).^2;
% pcolor(LON,LAT,C);shading interp;

figure;
m_proj('lambert','lon',LONLIMS,'lat',LATLIMS);

% m_pcolor(LO,LA,Rrs488);shading interp;
m_pcolor(LON,LAT,L865);shading interp;
% colormap(map);
m_coast('color','y');
% m_gshhs_i('patch',[0.9 0.8 0.7]);
% colormap gray;
% colormap jet;
% colormap (flipud(colormap(jet)));
m_grid('linewi',2,'tickdir','out');
h=colorbar;
title(HFil);
% 
% figure;
% imagesc(L865);



