package com.a504.chatcomposer.produce.dto.response;

import com.a504.chatcomposer.global.util.BaseResponseBody;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class CoverUrlResp extends BaseResponseBody {
    private String coverSource;

    public static CoverUrlResp of(String msg, Integer statusCode, String coverSource) {
        CoverUrlResp res = new CoverUrlResp();
        res.setMsg(msg);
        res.setStatusCode(statusCode);
        res.setCoverSource(coverSource);
        return res;
    }

}
