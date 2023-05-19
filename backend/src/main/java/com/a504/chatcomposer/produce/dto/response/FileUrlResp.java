package com.a504.chatcomposer.produce.dto.response;

import com.a504.chatcomposer.global.util.BaseResponseBody;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class FileUrlResp extends BaseResponseBody {
    private String source;

    public static FileUrlResp of(String msg, Integer statusCode, String coverSource) {
        FileUrlResp res = new FileUrlResp();
        res.setMsg(msg);
        res.setStatusCode(statusCode);
        res.setSource(coverSource);
        return res;
    }

}
