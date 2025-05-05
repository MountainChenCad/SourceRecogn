function test(CheckedNodes)
    [X,tf] = str2num("CheckedNodes");
    if tf = 0
        msgbox("采样率的输入应为数数值")
    end
    if isempty(CheckedNodes)
        temp="请选择调制类型 "+datestr(clock,'yyyy-mm-dd-HH:MM:SS' );
        %app.TextArea_10.Value=temp; 
    else
        if CheckedNodes(1).Text=="调制信号类型选择"
            CheckedNodes(1)=[];
            for i=1:length(CheckedNodes)
                modulationTypes = categorical([modulationTypes,cellstr(CheckedNodes(i).Text)]);    
            end
        else
            for i=1:length(CheckedNodes)
                modulationTypes = categorical([modulationTypes,cellstr(CheckedNodes(i).Text)]);    
            end    
        end
    end 
end